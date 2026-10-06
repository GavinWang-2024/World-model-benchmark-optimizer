"""Encode prompts for Wan2.1 once, on CPU, and save the embeddings.

    python scripts/encode_prompts.py                       # the default physics-flavoured prompt set
    python scripts/encode_prompts.py --prompt "a ball bounces" --prompt "..."
    python scripts/encode_prompts.py --from-set worldoptbench/prompts/wan_set_16.json   # every prompt in a set file

New prompts are MERGED into the existing embeddings file; prompts already encoded are skipped.

Why: the text encoder (UMT5-XXL, ~5.7B parameters, ~11 GB in bf16, 22.7 GB on disk in
fp32) does not fit on a 12 GB GPU beside the transformer, and it is only needed to turn text
into embeddings. Encode here, save, and the video model runs without it loaded at all
(`WanVideo` takes the saved embeddings). The pipeline's own `encode_prompt` is used so the
embeddings are exactly what a normal run would compute (512-token padding, as the
pipeline's default).

CPU encoding is slow (minutes per prompt) but one-off. After this you can delete the
text_encoder files from the Hugging Face cache to reclaim ~23 GB.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
OUT = Path(__file__).resolve().parent.parent / "results" / "wan_prompts.pt"

DEFAULT_PROMPTS = [
    "A red ball rolls down a wooden ramp and bounces on the floor.",
    "Water is poured from a pitcher into a glass on a kitchen counter.",
    "A robot arm picks up a red cube from a table and places it into a bin.",
    "A car drives through a busy four-way intersection while pedestrians cross the street.",
]
NEGATIVE = "blurry, low quality, distorted, jittery, static, overexposed, watermark, text"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", action="append", help="a prompt to encode (repeatable); default: the built-in set")
    ap.add_argument("--from-set", type=Path, help="encode every prompt in a standard-set JSON file")
    ap.add_argument("--negative", default=NEGATIVE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max-sequence-length", type=int, default=512)
    args = ap.parse_args()
    if args.from_set:
        import json

        entries = json.loads(args.from_set.read_text(encoding="utf-8"))["prompts"]
        prompts = list(dict.fromkeys(e["prompt"] for e in entries))  # seeds repeat prompts; encode each once
    else:
        prompts = args.prompt or DEFAULT_PROMPTS

    embeddings: dict[str, torch.Tensor] = {}
    negative = None
    if args.out.exists():
        saved = torch.load(args.out, map_location="cpu")
        if saved.get("negative_text") != args.negative:
            raise SystemExit("existing embeddings used a different negative prompt; delete the file or pass the same --negative")
        embeddings, negative = dict(saved["prompts"]), saved["negative"]
    todo = [p for p in prompts if p not in embeddings]
    print(f"{len(prompts) - len(todo)} already encoded, {len(todo)} to do", flush=True)
    if not todo:
        return 0

    from diffusers import WanPipeline

    path = snapshot_download(REPO_ID, allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*"])
    # transformer and vae are not needed to encode text, so don't load them
    pipe = WanPipeline.from_pretrained(path, transformer=None, vae=None, torch_dtype=torch.bfloat16)

    for prompt in todo:
        start = time.time()
        with torch.no_grad():
            pos, neg = pipe.encode_prompt(
                prompt=prompt, negative_prompt=args.negative, do_classifier_free_guidance=True,
                max_sequence_length=args.max_sequence_length, device=torch.device("cpu"),
            )
        embeddings[prompt] = pos.cpu().contiguous()
        negative = neg.cpu().contiguous()
        print(f"encoded in {time.time() - start:.0f}s, shape {tuple(pos.shape)} {pos.dtype}: {prompt[:60]}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompts": embeddings, "negative": negative, "negative_text": args.negative}, args.out)
    print(f"saved {len(embeddings)} prompt embeddings -> {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
