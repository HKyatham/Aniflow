"""Backend-agnostic LLM transport for the narration pipeline.

Ollama and OpenAI-compatible servers (9router) disagree on URL prefix, request
shape and response envelope. Call sites build one logical request; the active
arm translates it. Selected by ``config.USE_NINE_ROUTER``.
"""

from __future__ import annotations

import requests

import config

DEFAULT_TIMEOUT = getattr(config, "OLLAMA_TIMEOUT", 120)


def raise_for_llm_status(resp):
    """Like resp.raise_for_status(), but keeps the server's own error message.

    Ollama answers failures with a JSON body such as
    {"error": "model requires more system memory (9.6 GiB) than is available"};
    plain raise_for_status() drops it and leaves only "500 Server Error".
    """
    if resp.status_code < 400:
        return
    detail = ""
    try:
        body = resp.json()
        err = body.get("error") if isinstance(body, dict) else None
        if isinstance(err, dict):  # OpenAI-style {"error": {"message": ...}}
            err = err.get("message")
        detail = str(err or "")
    except ValueError:
        detail = (resp.text or "").strip()[:300]
    msg = f"{resp.status_code} {resp.reason} from {resp.url}"
    if detail:
        msg += f" — {detail}"
    raise requests.HTTPError(msg, response=resp)


# ── GPU → CPU fallback for vision models ─────────────────────────
# On some GPU setups Ollama runs a vision model's image path incorrectly:
# qwen2.5vl answers any image with "@@@@@@…" until Ollama aborts with
# "token repeat limit reached" (the 500s seen in the pipeline), while the
# same model on CPU reads the image fine. When that happens we switch the
# model to CPU (num_gpu=0) for the rest of the session and retry.
# Force it up front with OLLAMA_CPU_MODELS="qwen2.5vl:3b,other:tag".
_cpu_only_models = {m.strip() for m in
                    getattr(config, "OLLAMA_CPU_MODELS", "").split(",") if m.strip()}


def _has_images(payload: dict) -> bool:
    if payload.get("images"):
        return True
    return any(m.get("images") for m in payload.get("messages") or [])


def _reply_text(data: dict) -> str:
    return (data.get("response") or (data.get("message") or {}).get("content") or "").strip()


def _is_gpu_garbage(text: str = "", error: str = "") -> bool:
    if "token repeat limit" in error:
        return True
    return len(text) >= 3 and set(text) == {"@"}


def ollama_post(url: str, payload: dict, timeout, headers: dict | None = None) -> dict:
    """POST to an Ollama /api/generate or /api/chat endpoint and return its JSON,
    retrying once on CPU if the GPU produced vision garbage."""
    model = payload.get("model", "")
    for attempt in (0, 1):
        body = payload
        req_timeout = timeout
        if model in _cpu_only_models:
            body = {**payload, "options": {**(payload.get("options") or {}), "num_gpu": 0}}
            # CPU inference is several times slower than GPU
            req_timeout = max(timeout or 0, getattr(config, "OLLAMA_TIMEOUT", 180))
        resp = requests.post(url, json=body, headers=headers, timeout=req_timeout)
        garbage = False
        if resp.status_code >= 400:
            try:
                err = str((resp.json() or {}).get("error", ""))
            except ValueError:
                err = resp.text or ""
            garbage = _is_gpu_garbage(error=err)
            if not garbage:
                raise_for_llm_status(resp)
            data = {}
        else:
            data = resp.json()
            garbage = _is_gpu_garbage(text=_reply_text(data))
        if not garbage:
            return data
        if attempt == 1 or model in _cpu_only_models or not _has_images(payload):
            if resp.status_code >= 400:
                raise_for_llm_status(resp)
            return data
        _cpu_only_models.add(model)
        print(f"[LLM] ⚠️ {model} returned garbage for an image on the GPU "
              f"(a known Ollama GPU issue). Running {model} on CPU from now on — "
              f"slower, but it reads images correctly. Set OLLAMA_CPU_MODELS={model} "
              f"to skip this first failed attempt.")
    return data


def _as_data_url(image, default_media_type: str = "image/jpeg") -> str:
    """Accept bare base64, a ``(b64, media_type)`` pair, or an existing URL."""
    media_type = default_media_type
    if isinstance(image, (tuple, list)):
        image, *rest = image
        if rest and rest[0]:
            media_type = rest[0]
    if not image:
        return ""
    if isinstance(image, str) and image.startswith(("data:", "http://", "https://")):
        return image
    return f"data:{media_type};base64,{image}"


def _as_base64(image) -> str:
    """Ollama wants bare base64 (or a data URL it can decode), never a tuple."""
    if isinstance(image, (tuple, list)):
        image = image[0] if image else ""
    return image or ""


class BaseTransport:
    kind = "base"

    def __init__(self, host: str, api_key: str = ""):
        self.host = (host or "").rstrip("/")
        self.api_key = api_key or ""

    @property
    def headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    @staticmethod
    def _post(url: str, payload: dict, headers: dict, timeout):
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        raise_for_llm_status(resp)
        return resp.json()

    @staticmethod
    def _get(url: str, headers: dict, timeout):
        resp = requests.get(url, headers=headers, timeout=timeout)
        raise_for_llm_status(resp)
        return resp.json()

    def chat(self, **kwargs) -> str:
        raise NotImplementedError

    def generate(self, **kwargs) -> str:
        raise NotImplementedError

    def list_models(self) -> list[str]:
        raise NotImplementedError

    def describe(self) -> dict:
        """Liveness + installed model names, independent of any chosen model."""
        try:
            return {"running": True, "models": self.list_models()}
        except Exception:
            return {"running": False, "models": []}


class OllamaTransport(BaseTransport):
    """Ollama's native API: /api/chat, /api/generate, /api/tags."""

    kind = "ollama"

    @staticmethod
    def _options(temperature, top_p, max_tokens, repeat_penalty, top_k=None) -> dict:
        options = {"temperature": temperature, "top_p": top_p}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if repeat_penalty is not None:
            options["repeat_penalty"] = repeat_penalty
        if top_k is not None:
            options["top_k"] = top_k
        return options

    def chat(self, model: str, prompt: str, images: list | None = None, *,
             system: str | None = None, temperature: float = 0.7, top_p: float = 0.9,
             max_tokens: int | None = None, repeat_penalty: float | None = None,
             top_k: int | None = None, keep_alive: str | None = None,
             timeout=DEFAULT_TIMEOUT) -> str:
        user = {"role": "user", "content": prompt}
        if images:
            user["images"] = [_as_base64(img) for img in images]
        messages = ([{"role": "system", "content": system}] if system else []) + [user]
        payload = {
            "model": model,
            "stream": False,
            "messages": messages,
            "options": self._options(temperature, top_p, max_tokens, repeat_penalty, top_k),
        }
        if keep_alive:
            payload["keep_alive"] = keep_alive
        data = ollama_post(f"{self.host}/api/chat", payload, timeout, self.headers)
        return (data.get("message") or {}).get("content", "").strip()

    def generate(self, model: str, prompt: str, images: list | None = None, *,
                 temperature: float = 0.7, top_p: float = 0.9,
                 max_tokens: int | None = None, repeat_penalty: float | None = None,
                 top_k: int | None = None, keep_alive: str | None = None,
                 timeout=DEFAULT_TIMEOUT) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": self._options(temperature, top_p, max_tokens, repeat_penalty, top_k),
        }
        if images:
            payload["images"] = [_as_base64(img) for img in images]
        if keep_alive:
            payload["keep_alive"] = keep_alive
        data = ollama_post(f"{self.host}/api/generate", payload, timeout, self.headers)
        return data.get("response", "").strip()

    def list_models(self) -> list[str]:
        data = self._get(f"{self.host}/api/tags", self.headers, 5)
        return [m["name"] for m in data.get("models", []) if m.get("name")]


class OpenAICompatTransport(BaseTransport):
    """Any OpenAI-compatible server, including 9router: /v1/chat/completions, /v1/models."""

    kind = "openai"

    def chat(self, model: str, prompt: str, images: list | None = None, *,
             system: str | None = None, temperature: float = 0.7, top_p: float = 0.9,
             max_tokens: int | None = None, repeat_penalty: float | None = None,
             top_k: int | None = None, keep_alive: str | None = None,
             timeout=DEFAULT_TIMEOUT) -> str:
        # top_k and keep_alive are Ollama-only; an OpenAI-compatible server has
        # no equivalent, so they are dropped rather than guessed at.
        urls = [_as_data_url(img) for img in (images or [])]
        urls = [u for u in urls if u]
        if urls:
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": u}} for u in urls
            ]
        else:
            content = prompt
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": content}
        ]
        payload = {
            "model": model,
            "stream": False,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if repeat_penalty is not None:
            # Ollama multiplies token scores (1.0 = off); OpenAI's
            # frequency_penalty is additive over 0..2. Closest sibling, not equal.
            payload["frequency_penalty"] = round(max(0.0, min(2.0, (repeat_penalty - 1.0) * 2)), 2)
        data = self._post(f"{self.host}/v1/chat/completions", payload, self.headers, timeout)
        choices = data.get("choices") or [{}]
        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):  # some servers echo the content-part array back
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return (content or "").strip()

    def generate(self, model: str, prompt: str, images: list | None = None, **kwargs) -> str:
        # No single-turn completion endpoint is assumed on the server, so a bare
        # prompt becomes a one-message chat — the same request shape as chat().
        return self.chat(model, prompt, images, **kwargs)

    def list_models(self) -> list[str]:
        data = self._get(f"{self.host}/v1/models", self.headers, 5)
        return [m["id"] for m in data.get("data", []) if m.get("id")]


def build_transport(host: str | None = None) -> BaseTransport:
    """Return the transport for the active backend."""
    if getattr(config, "USE_NINE_ROUTER", False):
        return OpenAICompatTransport(
            host or getattr(config, "NINE_ROUTER_HOST", "http://127.0.0.1:20128"),
            getattr(config, "NINE_ROUTER_API_KEY", ""),
        )
    return OllamaTransport(host or config.OLLAMA_HOST)


def as_transport(host_or_transport) -> BaseTransport:
    """Coerce a legacy host string, or pass a transport through unchanged.

    A bare host has always meant "an Ollama server" in this codebase, so it is
    honoured literally rather than re-resolved through config.
    """
    if isinstance(host_or_transport, BaseTransport):
        return host_or_transport
    if host_or_transport:
        return OllamaTransport(str(host_or_transport))
    return build_transport()
