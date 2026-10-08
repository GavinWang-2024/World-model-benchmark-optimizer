"""Reads a project-root `.env` file so secrets (the Hugging Face token) stay out of the code and out of git.

`.env` is plain `KEY=value` lines (`#` comments, optional quotes). Values already present in the real environment win,
so a token exported in the shell overrides the file. The Hugging Face token may be written as `token`, `hf_token` or
`HF_TOKEN`; it is exported as `HF_TOKEN` (what `huggingface_hub`, `transformers` and `diffusers` read), and
`HUGGING_FACE_HUB_TOKEN` is set to the same value for older versions.
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_HF_KEYS = ("HF_TOKEN", "HF_TOKEN".lower(), "token")


def parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_env(path: str | Path | None = None) -> dict[str, str]:
    """Loads `.env` (default: the repo root) into os.environ without overriding existing variables.

    Returns the parsed file (so callers can see what was there); a missing file is not an error.
    """
    env_path = Path(path) if path is not None else _ROOT / ".env"
    try:
        values = parse_env(env_path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    for key, value in values.items():
        if key not in _HF_KEYS:
            os.environ.setdefault(key, value)
    token = next((values[k] for k in _HF_KEYS if values.get(k)), None)
    if token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = token
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    return values


def hf_token() -> str | None:
    """The Hugging Face token from the environment or `.env`, or None."""
    load_env()
    return os.environ.get("HF_TOKEN") or None
