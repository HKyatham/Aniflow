"""
ElevenLabs TTS Engine - Cloud-based high-quality text-to-speech.
Requires an ElevenLabs API key.
"""
import os
import re
import time
import requests
import json
import config

ELEVENLABS_AVAILABLE = True  # Available if user provides API key

_ELEVENLABS_ID_RE = re.compile(r'^[A-Za-z0-9]{20}$')


def _api_error(resp) -> str:
    """ElevenLabs' own explanation from an error response, e.g.
    {"detail": {"status": "missing_permissions", "message": "..."}}."""
    try:
        detail = resp.json().get('detail')
    except ValueError:
        return (resp.text or '').strip()[:200]
    if isinstance(detail, dict):
        return ' — '.join(str(x) for x in (detail.get('status'), detail.get('message')) if x)
    return str(detail or '')


class ElevenLabsTTS:
    """ElevenLabs TTS engine for high-quality voice synthesis."""

    def __init__(self):
        # From ELEVENLABS_API_KEY (environment or .env); the settings UI can
        # still replace it at runtime via set_api_key()
        self.api_key = (getattr(config, 'ELEVENLABS_API_KEY', '') or '').strip() or None
        self.base_url = "https://api.elevenlabs.io/v1"
        self._voices_cache = None
        self._voices_error = None      # (time, message) of the last failed fetch

    def set_api_key(self, api_key: str):
        """Set the ElevenLabs API key."""
        self.api_key = api_key
        self._voices_cache = None
        self._voices_error = None

    @property
    def available(self) -> bool:
        """Check if ElevenLabs is available (has API key)."""
        return bool(self.api_key)

    def generate_speech(self, text: str, output_path: str,
                        voice: str = None, model: str = None,
                        stability: float = 0.5, similarity_boost: float = 0.75) -> dict:
        """
        Generate speech audio using ElevenLabs API.

        Args:
            text: Text to synthesize
            output_path: Path to save the audio file
            voice: ElevenLabs voice ID
            model: Model ID (eleven_monolingual_v1, eleven_multilingual_v2, etc.)
            stability: Stability setting (0.0-1.0)
            similarity_boost: Clarity setting (0.0-1.0)

        Returns:
            dict with audio_path, duration
        """
        if not self.api_key:
            raise RuntimeError("ElevenLabs API key not set. Please configure your API key in settings.")

        model = model or getattr(config, 'ELEVENLABS_MODEL', 'eleven_multilingual_v2')

        # A voice name from another engine (e.g. Edge's "en-US-GuyNeural")
        # isn't an ElevenLabs voice ID — ignore it rather than send it.
        if voice and not _ELEVENLABS_ID_RE.match(voice):
            voice = None
        voice = voice or (getattr(config, 'ELEVENLABS_VOICE_ID', '') or '').strip() or None
        if not voice:
            # Default to first available voice
            voices = self.get_voices()
            if not voices:
                reason = self._voices_error[1] if self._voices_error else "the voice list is empty"
                raise RuntimeError(
                    f"No ElevenLabs voice to use ({reason}). Pick a voice in the app, or set "
                    f"ELEVENLABS_VOICE_ID in .env to a voice ID from your ElevenLabs library.")
            voice = voices[0]['voice_id']

        # ElevenLabs API endpoint
        url = f"{self.base_url}/text-to-speech/{voice}"

        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }

        data = {
            "text": text,
            "model_id": model,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost
            }
        }

        try:
            response = requests.post(url, json=data, headers=headers, timeout=60)
            response.raise_for_status()

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save audio
            mp3_path = output_path.rsplit('.', 1)[0] + '.mp3'
            with open(mp3_path, 'wb') as f:
                f.write(response.content)

            # Get duration
            duration = self._get_audio_duration(mp3_path)

            return {
                "audio_path": mp3_path,
                "duration": duration,
            }

        except requests.exceptions.HTTPError as e:
            code = e.response.status_code
            why = _api_error(e.response)
            if code == 401:
                raise RuntimeError(f"ElevenLabs rejected the API key (401): {why or 'invalid key'}")
            elif code == 403:
                raise RuntimeError(f"ElevenLabs quota exceeded or access denied (403): {why}")
            elif code == 422:
                raise RuntimeError(f"Invalid request to ElevenLabs API (422): {why}")
            else:
                raise RuntimeError(f"ElevenLabs API error ({code}): {why or e}")
        except requests.exceptions.Timeout:
            raise RuntimeError("ElevenLabs API request timed out")
        except Exception as e:
            raise RuntimeError(f"ElevenLabs TTS error: {e}")

    def get_voices(self, force_refresh: bool = False) -> list[dict]:
        """
        Get available ElevenLabs voices.
        
        Returns list of voice dicts with: voice_id, name, preview_url, labels, etc.
        """
        if not self.api_key:
            return []

        # Use cache unless force refresh
        if self._voices_cache is not None and not force_refresh:
            return self._voices_cache
        # Don't hammer the API (once per panel) after it just refused us
        if (not force_refresh and self._voices_error
                and time.monotonic() - self._voices_error[0] < 60):
            return []

        url = f"{self.base_url}/voices"
        headers = {"xi-api-key": self.api_key}

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code >= 400:
                msg = f"{response.status_code} from /v1/voices: {_api_error(response) or response.reason}"
                self._voices_error = (time.monotonic(), msg)
                print(f"[ElevenLabs] Error fetching voices: {msg}")
                return []
            data = response.json()

            voices = []
            for v in data.get('voices', []):
                voices.append({
                    "voice_id": v['voice_id'],
                    "name": v['name'],
                    "preview_url": v.get('preview_url'),
                    "category": v.get('category', 'generated'),
                    "labels": v.get('labels', {}),
                    "description": v.get('description', ''),
                    "engine": "elevenlabs",
                })

            self._voices_cache = voices
            self._voices_error = None
            return voices

        except Exception as e:
            self._voices_error = (time.monotonic(), str(e))
            print(f"[ElevenLabs] Error fetching voices: {e}")
            return []

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        try:
            from moviepy.editor import AudioFileClip
            clip = AudioFileClip(audio_path)
            duration = clip.duration
            clip.close()
            return duration
        except Exception:
            # Fallback: estimate from file size (rough approximation)
            try:
                size = os.path.getsize(audio_path)
                # MP3 at ~128kbps: 1 second ≈ 16KB
                return max(1.0, size / 16000)
            except Exception:
                return 2.0

    def test_api_key(self) -> tuple[bool, str]:
        """
        Test if the API key is valid.
        
        Returns:
            (success: bool, message: str)
        """
        if not self.api_key:
            return False, "No API key provided"

        try:
            voices = self.get_voices(force_refresh=True)
            if voices:
                return True, f"API key valid. {len(voices)} voices available."
            if self._voices_error:
                return False, f"ElevenLabs refused the key: {self._voices_error[1]}"
            return False, "API key valid but no voices found"
        except Exception as e:
            return False, str(e)


# Singleton
_elevenlabs_tts = None


def get_elevenlabs_tts() -> ElevenLabsTTS:
    """Get the global ElevenLabs TTS instance."""
    global _elevenlabs_tts
    if _elevenlabs_tts is None:
        _elevenlabs_tts = ElevenLabsTTS()
    return _elevenlabs_tts
