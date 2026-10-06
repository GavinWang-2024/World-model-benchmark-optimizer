"""Download the Wan2.1-T2V-1.3B diffusers files we need into the Hugging Face cache.

    python scripts/fetch_wan.py                      # everything needed (~29 GB, mostly the text encoder)
    python scripts/fetch_wan.py --skip-text-encoder  # ~6.2 GB: transformer, VAE, tokenizer, scheduler

Skips the repo's demo assets/examples. The model is public (Apache 2.0), so no token
is needed. The text encoder (UMT5-XXL, ~22.7 GB in fp32) is only needed once, to turn
prompts into embeddings (`scripts/encode_prompts.py`); it does not fit on a 12 GB GPU
beside the transformer, so after encoding it can be deleted from the cache.

Re-running resumes: files already in the cache are not downloaded again.
"""

from __future__ import annotations

import argparse
import time

from huggingface_hub import snapshot_download

REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
CORE = ["model_index.json", "scheduler/*", "tokenizer/*", "transformer/*", "vae/*"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-text-encoder", action="store_true")
    args = ap.parse_args()

    patterns = CORE + ([] if args.skip_text_encoder else ["text_encoder/*"])
    print(f"fetching {REPO_ID}: {patterns}", flush=True)
    start = time.time()
    path = snapshot_download(REPO_ID, allow_patterns=patterns)
    print(f"done in {time.time() - start:.0f}s -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
