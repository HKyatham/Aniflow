"""
ElevenLabs TTS Engine - Cloud-based high-quality text-to-speech.

Uses the official ElevenLabs Python SDK (`pip install elevenlabs`):

    client = ElevenLabs(api_key=...)
    audio = client.text_to_speech.convert(voice_id=..., text=..., model_id=...)

Configuration (environment, .env or .env.local):
    ELEVENLABS_API_KEY   required
    ELEVENLABS_VOICE_ID  voice used when none is picked in the app
    ELEVENLABS_MODEL     default "eleven_v4" (supports audio tags such as
                         "[whisper]" or "[building tension]" in the text)
"""
import os
import re
import time

import config

try:
    from elevenlabs import VoiceSettings
    from elevenlabs.client import ElevenLabs
    from elevenlabs.core.api_error import ApiError
    ELEVENLABS_AVAILABLE = True
except ImportError:  # SDK not installed — engine reports itself unavailable
    ElevenLabs = VoiceSettings = None

    class ApiError(Exception):
        status_code = None
        body = None

    ELEVENLABS_AVAILABLE = False

# ElevenLabs voice IDs are 20 alphanumeric characters (e.g. kLhAstPcnnPxqzk6gS5i)
_ELEVENLABS_ID_RE = re.compile(r'^[A-Za-z0-9]{20}$')


def _api_error_detail(e) -> str:
    """ElevenLabs' own explanation, e.g. "missing_permissions — The API key
    you used is missing the permission voices_read ..." """
    body = getattr(e, 'body', None)
    detail = body.get('detail', body) if isinstance(body, dict) else body
    if isinstance(detail, dict):
        return ' — '.join(str(x) for x in (detail.get('status'), detail.get('message')) if x)
    return str(detail or e)


class ElevenLabsTTS:
    """ElevenLabs TTS engine for high-quality voice synthesis."""

    def __init__(self):
        self._client = None
        self._voices_cache = None
        self._voices_error = None  # (time, message) of the last failed voice fetch
        self.api_key = None
        # From ELEVENLABS_API_KEY (environment or .env); the settings UI can
        # still replace it at runtime via set_api_key()
        self.set_api_key((getattr(config, 'ELEVENLABS_API_KEY', '') or '').strip() or None)

    def set_api_key(self, api_key: str):
        """Set the ElevenLabs API key (and rebuild the SDK client)."""
        self.api_key = (api_key or '').strip() or None
        self._client = (ElevenLabs(api_key=self.api_key)
                        if (self.api_key and ELEVENLABS_AVAILABLE) else None)
        self._voices_cache = None
        self._voices_error = None

    @property
    def available(self) -> bool:
        """True when the SDK is installed and an API key is set."""
        return self._client is not None

    def _require_client(self):
        if not ELEVENLABS_AVAILABLE:
            raise RuntimeError("The ElevenLabs SDK is not installed. Run: pip install elevenlabs")
        if self._client is None:
            raise RuntimeError("ElevenLabs API key not set. Add ELEVENLABS_API_KEY to .env "
                               "or enter it in settings.")
        return self._client

    def _pick_voice(self, voice: str = None) -> str:
        # A voice name from another engine (e.g. Edge's "en-US-GuyNeural")
        # isn't an ElevenLabs voice ID — ignore it rather than send it.
        if voice and not _ELEVENLABS_ID_RE.match(voice):
            voice = None
        voice = voice or (getattr(config, 'ELEVENLABS_VOICE_ID', '') or '').strip() or None
        if voice:
            return voice
        voices = self.get_voices()
        if voices:
            return voices[0]['voice_id']
        reason = self._voices_error[1] if self._voices_error else "the voice list is empty"
        raise RuntimeError(
            f"No ElevenLabs voice to use ({reason}). Pick a voice in the app, or set "
            f"ELEVENLABS_VOICE_ID in .env to a voice ID from your ElevenLabs library.")

    def generate_speech(self, text: str, output_path: str,
                        voice: str = None, model: str = None,
                        stability: float = None, similarity_boost: float = None) -> dict:
        """
        Generate speech audio and save it as MP3.

        Args:
            text: Text to synthesize (may include audio tags like "[whisper]"
                  with models that support them, e.g. eleven_v4)
            output_path: Where to save; the extension is forced to .mp3
            voice: ElevenLabs voice ID (falls back to ELEVENLABS_VOICE_ID)
            model: Model ID (falls back to ELEVENLABS_MODEL)
            stability / similarity_boost: optional voice settings; omitted by
                  default so the voice's own saved settings apply

        Returns:
            dict with audio_path, duration
        """
        client = self._require_client()
        voice_id = self._pick_voice(voice)
        model_id = model or getattr(config, 'ELEVENLABS_MODEL', 'eleven_v4')

        kwargs = {'voice_id': voice_id, 'text': text, 'model_id': model_id,
                  'output_format': 'mp3_44100_128'}
        if stability is not None or similarity_boost is not None:
            kwargs['voice_settings'] = VoiceSettings(stability=stability,
                                                     similarity_boost=similarity_boost)

        try:
            audio = b"".join(client.text_to_speech.convert(**kwargs))
        except ApiError as e:
            code, why = e.status_code, _api_error_detail(e)
            if code == 401:
                raise RuntimeError(f"ElevenLabs rejected the API key (401): {why}")
            if code == 403:
                raise RuntimeError(f"ElevenLabs quota exceeded or access denied (403): {why}")
            if code == 404:
                raise RuntimeError(f"ElevenLabs voice or model not found (404, voice={voice_id}, "
                                   f"model={model_id}): {why}")
            if code == 422:
                raise RuntimeError(f"Invalid request to ElevenLabs (422): {why}")
            raise RuntimeError(f"ElevenLabs API error ({code}): {why}")
        except Exception as e:
            raise RuntimeError(f"ElevenLabs TTS error: {e}")

        if not audio:
            raise RuntimeError("ElevenLabs returned empty audio")

        mp3_path = os.path.splitext(output_path)[0] + '.mp3'
        os.makedirs(os.path.dirname(mp3_path) or '.', exist_ok=True)
        with open(mp3_path, 'wb') as f:
            f.write(audio)

        return {
            "audio_path": mp3_path,
            "duration": self._get_audio_duration(mp3_path),
        }

    def get_voices(self, force_refresh: bool = False) -> list[dict]:
        """
        Get the voices in the account's ElevenLabs library.

        Returns list of voice dicts with: voice_id, name, preview_url, labels, etc.
        Needs the API key's "Voices: Read" permission.
        """
        if not self.available:
            return []
        if self._voices_cache is not None and not force_refresh:
            return self._voices_cache
        # Don't hammer the API (once per panel) right after it refused us
        if (not force_refresh and self._voices_error
                and time.monotonic() - self._voices_error[0] < 60):
            return []

        try:
            voices, token = [], None
            while True:
                resp = self._client.voices.search(page_size=100, next_page_token=token)
                for v in resp.voices or []:
                    voices.append({
                        "voice_id": v.voice_id,
                        "name": v.name,
                        "preview_url": getattr(v, 'preview_url', None),
                        "category": getattr(v, 'category', None) or 'generated',
                        "labels": getattr(v, 'labels', None) or {},
                        "description": getattr(v, 'description', None) or '',
                        "engine": "elevenlabs",
                    })
                token = getattr(resp, 'next_page_token', None)
                if not getattr(resp, 'has_more', False) or not token:
                    break
            self._voices_cache = voices
            self._voices_error = None
            return voices
        except ApiError as e:
            msg = f"{e.status_code} from voice list: {_api_error_detail(e)}"
        except Exception as e:
            msg = str(e)
        self._voices_error = (time.monotonic(), msg)
        print(f"[ElevenLabs] Error fetching voices: {msg}")
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
            # Fallback: estimate from file size (MP3 at 128kbps ≈ 16KB/s)
            try:
                return max(1.0, os.path.getsize(audio_path) / 16000)
            except Exception:
                return 2.0

    def test_api_key(self) -> tuple[bool, str]:
        """
        Test if the API key works.

        Returns:
            (success: bool, message: str)
        """
        if not ELEVENLABS_AVAILABLE:
            return False, "The ElevenLabs SDK is not installed. Run: pip install elevenlabs"
        if not self.api_key:
            return False, "No API key provided"

        voices = self.get_voices(force_refresh=True)
        if voices:
            return True, f"API key valid. {len(voices)} voices available."
        if self._voices_error:
            if getattr(config, 'ELEVENLABS_VOICE_ID', '') and 'voices_read' in self._voices_error[1]:
                return True, ("API key is set but can't list voices (missing the Voices: Read "
                              "permission), so ELEVENLABS_VOICE_ID will be used for speech.")
            return False, f"ElevenLabs refused the key: {self._voices_error[1]}"
        return False, "API key valid but no voices found"


    def status(self) -> dict:
        """What the app currently sees — for logs and troubleshooting.
        Never includes the full key."""
        key = self.api_key or ''
        return {
            'sdk_installed': ELEVENLABS_AVAILABLE,
            'api_key': f"set (…{key[-4:]})" if key else "NOT SET",
            'voice_id': (getattr(config, 'ELEVENLABS_VOICE_ID', '') or '').strip() or "not set",
            'model': getattr(config, 'ELEVENLABS_MODEL', 'eleven_v4'),
            'ready': self.available,
        }


# Singleton
_elevenlabs_tts = None


def get_elevenlabs_tts() -> ElevenLabsTTS:
    """Get the global ElevenLabs TTS instance."""
    global _elevenlabs_tts
    if _elevenlabs_tts is None:
        _elevenlabs_tts = ElevenLabsTTS()
    return _elevenlabs_tts


def _self_test():
    """python -m pipeline.elevenlabs_tts — check the setup end to end:
    shows what the app sees, lists voices with their IDs, writes a test MP3."""
    import tempfile
    tts = get_elevenlabs_tts()
    print("ElevenLabs setup as the app sees it:")
    for k, v in tts.status().items():
        print(f"  {k:14} {v}")
    if not ELEVENLABS_AVAILABLE:
        print("\n✗ Install the SDK in the same Python environment the app uses:  pip install elevenlabs")
        return
    if not tts.api_key:
        print(f"\n✗ No key found. Put ELEVENLABS_API_KEY=... in {os.path.join(config.BASE_DIR, '.env')}"
              f" (check it isn't saved as .env.txt), or set it in this terminal.")
        return

    print("\nVoices in your account:")
    voices = tts.get_voices(force_refresh=True)
    for v in voices[:25]:
        print(f"  {v['voice_id']}  {v['name']}  ({v['category']})")
    if not voices:
        print(f"  (none listed — {tts._voices_error[1] if tts._voices_error else 'empty library'})")

    out = os.path.join(tempfile.gettempdir(), "aniflow_elevenlabs_test.mp3")
    print("\nGenerating a test clip…")
    try:
        r = tts.generate_speech("[building tension] Nobody wanted to. [whisper] But everybody knew.", out)
        print(f"✓ Saved {r['audio_path']} ({r['duration']:.1f}s) — ElevenLabs works.")
        if not (getattr(config, 'ELEVENLABS_VOICE_ID', '') or '').strip() and voices:
            print(f"  Used the first voice ({voices[0]['name']}). To pick one, add to .env:\n"
                  f"  ELEVENLABS_VOICE_ID={voices[0]['voice_id']}")
    except RuntimeError as e:
        print(f"✗ {e}")


if __name__ == "__main__":
    _self_test()
