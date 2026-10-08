"""Side-by-side contact sheets of Wan videos: the baseline, the numerical-perturbation control and optimized
configurations, for a few prompts, so a person can check whether the reference-free "style deviation" ranking
matches what the videos look like.

    python scripts/make_contact_sheets.py                       # default prompts and configuration groups
    python scripts/make_contact_sheets.py --prompts ball_ramp_s0 pendulum_s0

Writes results/contact_sheets/<prompt>_<group>.png: one row per configuration (label: speedup and style deviation
from the 32-clip sweeps in results/wan_all, when present), six evenly spaced frames per row. Two groups per prompt:
`gentle` (up to about 1.5x) and `aggressive` (about 1.65x and up); both start with the baseline and the
perturbation control, which shows how much a numerically equivalent change alone moves the video.

Videos are regenerated here (the sweeps do not keep them), one model per configuration; ~10-15 min for the defaults.
Keep the GPU otherwise idle.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_wan import CONFIGS, MODEL_KWARGS, _mag_ratios, _perturbation

SET16 = ROOT / "worldoptbench" / "prompts" / "wan_set_16.json"

DEFAULT_PROMPTS = ["ball_ramp_s0", "water_pour_s0", "pendulum_s0", "pan_forest_s0", "skateboard_s0", "candle_flame_s0"]
GROUPS = {
    "gentle": ["baseline", "perturb_1e-4", "cfg_trunc_0.6", "wc_0.02", "fbc_0.05", "cfg0.4+pab_2", "taylorseer_3", "steps_20"],
    "aggressive": ["baseline", "perturb_1e-4", "wc_0.04", "fbc_0.1", "ada_slow30", "cfg0.6+wc_0.04",
                   "ada_slow30_x0.5", "wc_0.08", "ada_fast30"],
    "magcache": ["baseline", "perturb_1e-4", "wc_0.04", "kv+vae_bf16+cfg0.4+pab", "mag_0.24", "mag_0.24+cfg0.6+kv+vae",
                 "mag_0.5+cfg0.6+kv+vae", "mag_0.5_skip5", "mag_0.24_ret0.1"],
}
FRAMES_SHOWN = 6


def stem(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def measured(results_dir: Path, name: str) -> tuple[float | None, float | None]:
    """(mean speedup, mean style deviation) of a configuration from its sweep JSON, if there is one."""
    if name == "baseline":
        return 1.0, 0.0
    path = results_dir / f"{stem(name)}.json"
    if not path.exists():
        return None, None
    rows = json.loads(path.read_text(encoding="utf-8"))
    speed = st.mean(r["speed"]["speedup"] for r in rows)
    devs = [r["visual"].get("style_deviation") for r in rows if r["visual"].get("style_deviation") is not None]
    return speed, (st.mean(devs) if devs else None)


def generate(names: set[str], prompts: dict[str, dict]) -> dict[tuple[str, str], np.ndarray]:
    from worldoptbench.models.wan_video import WanVideo
    from worldoptbench.stack import OptimizationStack

    videos: dict[tuple[str, str], np.ndarray] = {}
    for name in sorted(names):
        modules, kwargs = CONFIGS[name]
        model = WanVideo(**MODEL_KWARGS)
        if "magcache" in modules:  # MagCache needs ratios measured on this setup (cached after the first calibration)
            kwargs = {**kwargs, "magcache": {**kwargs["magcache"], "mag_ratios": _mag_ratios(model)}}
        stack = OptimizationStack(model, modules, module_kwargs=kwargs)
        if stack.skipped:
            print(f"   {name}: skipped {[s.name for s in stack.skipped]}", flush=True)
        stack.apply()
        if name.startswith("perturb_"):
            model.step_callbacks.append(_perturbation(float(name.split("_", 1)[1])))
        for prompt_id, entry in prompts.items():
            frames = model.generate(prompt=entry["prompt"], horizon=2, seed=int(entry.get("seed", 0))).frames
            videos[(name, prompt_id)] = np.stack(frames)
        print(f"generated {name}", flush=True)
        stack.restore()
        del model, stack
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    return videos


def sheet(rows: list[tuple[str, np.ndarray, str]], title: str):
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.load_default(size=15)
        small = ImageFont.load_default(size=13)
    except TypeError:  # older Pillow: fixed-size default font
        font = small = ImageFont.load_default()
    height, width = rows[0][1].shape[1:3]
    label_w, pad, header = 230, 4, 26
    canvas = Image.new("RGB", (label_w + FRAMES_SHOWN * (width + pad), header + len(rows) * (height + pad)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title, fill=(255, 255, 255), font=font)
    for r, (name, video, caption) in enumerate(rows):
        top = header + r * (height + pad)
        draw.text((6, top + 6), name, fill=(255, 220, 120), font=font)
        draw.text((6, top + 30), caption, fill=(200, 200, 200), font=small)
        for c, index in enumerate(np.linspace(0, len(video) - 1, FRAMES_SHOWN).astype(int)):
            canvas.paste(Image.fromarray(np.ascontiguousarray(video[index])), (label_w + c * (width + pad), top))
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS, help="ids from wan_set_16.json")
    ap.add_argument("--groups", nargs="*", default=list(GROUPS), choices=list(GROUPS))
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "wan_all")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "contact_sheets")
    args = ap.parse_args()

    entries = {e["id"]: e for e in json.loads(SET16.read_text(encoding="utf-8"))["prompts"]}
    missing = [p for p in args.prompts if p not in entries]
    if missing:
        raise SystemExit(f"unknown prompt ids {missing}; see {SET16}")
    prompts = {p: entries[p] for p in args.prompts}
    names = {n for g in args.groups for n in GROUPS[g]}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    videos = generate(names, prompts)
    for prompt_id, entry in prompts.items():
        for group in args.groups:
            rows = []
            for name in GROUPS[group]:
                speed, dev = measured(args.results_dir, name)
                caption = ("" if speed is None else f"{speed:.2f}x") + ("" if dev is None else f"   style dev {dev:.3f}")
                if name.startswith("perturb_"):
                    caption = "noise control (no speedup)" + (f"   style dev {dev:.3f}" if dev is not None else "")
                rows.append((name, videos[(name, prompt_id)], caption.strip()))
            image = sheet(rows, f"{prompt_id}: {entry['prompt']}  [{group}]")
            path = args.out_dir / f"{prompt_id}_{group}.png"
            image.save(path)
            print("wrote", path, image.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
