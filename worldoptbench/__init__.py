"""WorldOptBench — architecture-agnostic benchmarking and optimization for world model inference."""

from worldoptbench.env import load_env

__version__ = "0.0.1"

load_env()  # exports HF_TOKEN from the project's .env (gitignored) if there is one, so gated downloads just work
