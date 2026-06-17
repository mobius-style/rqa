"""Local adapter client (Ollama) — the frozen-base generator."""
from __future__ import annotations

import requests


class AdapterError(RuntimeError):
    pass


class OllamaAdapter:
    def __init__(self, model: str, url: str, num_ctx: int, temperature: float):
        self.model = model
        self.url = url.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature

    def _post_chat(self, payload: dict, stream: bool = False):
        """POST /api/chat with think=false; retry without it for models that
        reject the parameter. Thinking mode burns the whole token budget on a
        `thinking` field and returns empty content (observed on qwen3.5 AND on
        gemma4-derived models after an Ollama update)."""
        payload = dict(payload)
        payload["think"] = False
        try:
            resp = requests.post(f"{self.url}/api/chat", json=payload, stream=stream, timeout=600)
            if resp.status_code == 400 and b"think" in resp.content:
                payload.pop("think", None)
                resp = requests.post(f"{self.url}/api/chat", json=payload, stream=stream, timeout=600)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            raise AdapterError(f"ollama request failed: {exc}") from exc

    def chat(self, system: str, messages: list[dict], json_mode: bool = True) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "num_predict": 8192 if json_mode else 1024,
            },
        }
        if json_mode:
            payload["format"] = "json"
        resp = self._post_chat(payload)
        data = resp.json()
        msg = data.get("message") or {}
        content = msg.get("content", "")
        if not content.strip():
            hint = " (thinking-mode leak?)" if msg.get("thinking") else ""
            raise AdapterError(f"empty adapter response (model={self.model}){hint}")
        return content

    def stream_chat(self, system: str, messages: list[dict]):
        """Yield content chunks from a streaming chat completion (no JSON mode)."""
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": True,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "num_predict": 1024,
            },
        }
        try:
            with self._post_chat(payload, stream=True) as resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    import json as _json

                    data = _json.loads(line)
                    chunk = (data.get("message") or {}).get("content", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break
        except requests.RequestException as exc:
            raise AdapterError(f"ollama stream failed: {exc}") from exc

    def health(self) -> bool:
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=5)
            resp.raise_for_status()
            return any(m.get("name") == self.model for m in resp.json().get("models", []))
        except requests.RequestException:
            return False
