from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ModelAdapter(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, grid: Any | None = None) -> str:
        raise NotImplementedError


class TransformersAdapter(ModelAdapter):
    """Lazy local Hugging Face adapter for Kaggle-attached models.

    Internet is never required. The model directory must already exist on disk.
    Text-only operation is intentional for v0.1: the policy supplies object-centric
    scene summaries and selective local crops. A vision adapter can be plugged into
    the same interface without changing the experiment harness.
    """

    def __init__(self, model_path: str, max_new_tokens: int = 512) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
        )
        self.max_new_tokens = max_new_tokens

    def complete(self, system: str, user: str, grid: Any | None = None) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new = out[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new, skip_special_tokens=True)


def extract_json(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


class OpenAICompatLocalAdapter(ModelAdapter):
    """Talk to a local OpenAI-compatible inference server such as vLLM.

    The endpoint is deliberately localhost-only by default, keeping Kaggle evaluation offline.
    """

    def __init__(
        self,
        model: str = "arc3",
        base_url: str = "http://127.0.0.1:8000/v1",
        max_tokens: int = 512,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, system: str, user: str, grid: Any | None = None) -> str:
        import base64
        import io
        import requests

        user_content: Any = user
        if grid is not None:
            try:
                import numpy as np
                from PIL import Image

                palette = np.array([
                    [255,255,255],[204,204,204],[153,153,153],[102,102,102],
                    [51,51,51],[0,0,0],[229,58,163],[255,123,204],
                    [249,60,49],[30,147,255],[136,216,241],[255,220,0],
                    [255,133,27],[146,18,49],[79,204,48],[163,86,214],
                ], dtype=np.uint8)
                arr = palette[np.asarray(grid, dtype=np.int64)]
                img = Image.fromarray(arr, mode="RGB").resize((256, 256), Image.Resampling.NEAREST)
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                user_content = [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]
            except Exception:
                user_content = user

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        r = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout)
        # If a text-only model rejects image content, retry with the same structured text.
        if not r.ok and grid is not None:
            payload["messages"][1]["content"] = user
            r = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return str(r.json()["choices"][0]["message"]["content"])
