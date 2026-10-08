"""Encode prompts for Cosmos-Predict2.5-2B once, on CPU, and save the embeddings.

    python scripts/encode_cosmos_prompts.py                       # the default physics-flavoured prompt set
    python scripts/encode_cosmos_prompts.py --prompt "a ball bounces" --prompt "..."
    python scripts/encode_cosmos_prompts.py --from-set worldoptbench/prompts/wan_set_16.json

New prompts are MERGED into the existing embeddings file; prompts already encoded are skipped.

Why: the Qwen2.5-VL-7B text encoder is ~16.6 GB in bf16, too big for a 12 GB GPU beside the transformer, and it is only
needed to turn text into embeddings. Encode here, save, and `CosmosPredict` runs without it loaded. The pipeline's own
`_get_prompt_embeds` is used, so the embeddings are what a normal run computes (chat template, 512-token padding, every
hidden layer normalised and concatenated: 512 x 100,352 per prompt, ~100 MB in bf16). The pipeline object itself is not
built (it would need the safety checker just to encode text); the unbound method is called on a small stand-in carrying
the tokenizer and the encoder.

CPU encoding takes a minute or two per prompt and needs ~17 GB of RAM. After this you can delete the text_encoder files
from the Hugging Face cache to reclaim ~16.6 GB.
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from huggingface_hub import snapshot_download

from worldoptbench.models.cosmos_predict import DEFAULT_EMBEDDINGS, REPO_ID, REVISION

DEFAULT_PROMPTS = [
    "A red ball rolls down a wooden ramp and bounces on the floor.",
    "Water is poured from a pitcher into a glass on a kitchen counter.",
    "A robot arm picks up a red cube from a table and places it into a bin.",
    "A car drives through a busy four-way intersection while pedestrians cross the street.",
]
# The pipeline's own default negative prompt, so unguided defaults match a stock run.
NEGATIVE = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
    "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
    "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
    "Overall, the video is of poor quality."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", action="append", help="a prompt to encode (repeatable); default: the built-in set")
    ap.add_argument("--from-set", type=Path, help="encode every prompt in a standard-set JSON file")
    ap.add_argument("--negative", default=NEGATIVE)
    ap.add_argument("--out", type=Path, default=DEFAULT_EMBEDDINGS)
    ap.add_argument("--max-sequence-length", type=int, default=512)
    args = ap.parse_args()
    if args.from_set:
        import json

        entries = json.loads(args.from_set.read_text(encoding="utf-8"))["prompts"]
        prompts = list(dict.fromkeys(e["prompt"] for e in entries))
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
    todo_negative = negative is None
    print(f"{len(prompts) - len(todo)} already encoded, {len(todo)} to do", flush=True)
    if not todo and not todo_negative:
        return 0

    from diffusers import Cosmos2_5_PredictBasePipeline
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    path = snapshot_download(REPO_ID, revision=REVISION, allow_patterns=["tokenizer/*", "text_encoder/*"])
    tokenizer = AutoTokenizer.from_pretrained(path, subfolder="tokenizer")
    encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    encoder.eval()
    cpu = torch.device("cpu")
    stand_in = types.SimpleNamespace(tokenizer=tokenizer, text_encoder=encoder, _execution_device=cpu)

    def encode(text: str) -> torch.Tensor:
        with torch.no_grad():
            return Cosmos2_5_PredictBasePipeline._get_prompt_embeds(
                stand_in, prompt=text, max_sequence_length=args.max_sequence_length, device=cpu, dtype=torch.bfloat16
            ).cpu().contiguous()

    if todo_negative:
        negative = encode(args.negative)
        print(f"encoded the negative prompt, shape {tuple(negative.shape)}", flush=True)
    for prompt in todo:
        start = time.time()
        embeddings[prompt] = encode(prompt)
        print(f"encoded in {time.time() - start:.0f}s, shape {tuple(embeddings[prompt].shape)}: {prompt[:60]}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompts": embeddings, "negative": negative, "negative_text": args.negative}, args.out)
    print(f"saved {len(embeddings)} prompt embeddings -> {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
