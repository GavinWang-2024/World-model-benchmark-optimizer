"""Measure optimizations on V-JEPA 2 (ViT-L, facebook/vjepa2-vitl-fpc64-256) as a latent forecaster.

    python scripts/sweep_vjepa2.py                       # all configurations, 10 clips
    python scripts/sweep_vjepa2.py --only baseline bf16  # a subset (baseline always runs first)

V-JEPA 2 predicts representations, so there is no PSNR/physics here. The task is the model's own: encode a context
clip, forecast the encoder latents of what follows (`worldoptbench/models/vjepa2.py`), and score the forecast in latent
space (`worldoptbench/metrics/latent.py`):

  * fidelity to the truth: the unoptimized fp32 encoder's latents of the real future (the full 64-frame clip), with a
    persistence baseline ("repeat the last context step") and a skill score against it;
  * agreement with the unoptimized fp32 forecast for the same clip (what the optimization changed).

Data: real video, 10 public Kinetics clips (`nateraw/kinetics-mini`: 2 each of archery, bowling, flying_kite,
high_jump, marching), 64 frames sampled evenly across each ~10 s clip. The first 32 frames are the context, the model
forecasts 16 tubelet-steps (the next 32 frames). The clips are downloaded to the Hugging Face cache and decoded to
`results/vjepa2_clips/`; neither is committed. The frame spacing is ~0.16 s (the model was trained at 4 fps, 0.25 s), so
this is not the training regime; every configuration sees the same input.

Timing is the model only: the input is prepared once (the CPU resize is a constant ~50 ms, reported separately),
every configuration is warmed up before timing (the first call of a compiled model pays its compile, reported as
`warmup_s`), and each clip is timed `--repeats` times (median). Each configuration runs in its own process because
TF32 and cudnn flags are process-wide. Ten clips from five classes is a screen, not a benchmark: effects smaller than the
clip-to-clip spread are not demonstrated.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET = "nateraw/kinetics-mini"
CLIPS_PER_CLASS = 2
N_FRAMES, CONTEXT_FRAMES = 64, 32
CLIP_DIR = ROOT / "results" / "vjepa2_clips"
REF_DIR = ROOT / "results" / "vjepa2_refs"

_COMPILE_TARGETS = {"torch_compile": {"targets": ("encoder", "predictor")}}

# name -> (modules in order, per-module kwargs, model kwargs)
CONFIGS: dict[str, tuple[list[str], dict[str, dict], dict]] = {
    "baseline": ([], {}, {}),
    "baseline (repeat)": ([], {}, {}),
    "tf32": (["tf32"], {}, {}),
    "bf16 autocast": (["precision"], {"precision": {"dtype": "bfloat16"}}, {}),
    "fp16 autocast": (["precision"], {"precision": {"dtype": "float16"}}, {}),
    "bf16 weights": ([], {}, {"torch_dtype": "bfloat16"}),
    "int8 weight-only": (["quantization"], {"quantization": {"scheme": "int8_weight_only"}}, {}),
    "int8 dynamic": (["quantization"], {"quantization": {"scheme": "int8_dynamic"}}, {}),
    "bf16 weights + fp8 weight-only": (
        ["quantization"], {"quantization": {"scheme": "float8_weight_only"}}, {"torch_dtype": "bfloat16"}),
    "bf16 weights + int8 weight-only": (
        ["quantization"], {"quantization": {"scheme": "int8_weight_only"}}, {"torch_dtype": "bfloat16"}),
    "compile": (["torch_compile"], _COMPILE_TARGETS, {}),
    "bf16 autocast + compile": (
        ["precision", "torch_compile"], {"precision": {"dtype": "bfloat16"}, **_COMPILE_TARGETS}, {}),
}


# ---- data ---------------------------------------------------------------------------------------------


def select_clips(n_per_class: int = CLIPS_PER_CLASS) -> list[str]:
    """Deterministic: the first `n_per_class` files (sorted by name) of each class in the dataset."""
    from huggingface_hub import list_repo_files

    by_class: dict[str, list[str]] = {}
    for f in sorted(list_repo_files(DATASET, repo_type="dataset")):
        if f.endswith(".mp4"):
            by_class.setdefault(f.split("/")[1], []).append(f)
    return [f for cls in sorted(by_class) for f in by_class[cls][:n_per_class]]


def load_clip(repo_file: str) -> np.ndarray:
    """(64, H, W, 3) uint8: 64 frames sampled evenly across the video (decoded once, cached as .npy)."""
    cache = CLIP_DIR / (repo_file.replace("/", "__").removesuffix(".mp4") + ".npy")
    if cache.exists():
        return np.load(cache)
    import av
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, repo_file, repo_type="dataset")
    with av.open(path) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    if len(frames) < N_FRAMES:
        raise ValueError(f"{repo_file} has only {len(frames)} frames; need {N_FRAMES}")
    clip = np.stack([frames[i] for i in np.linspace(0, len(frames) - 1, N_FRAMES).round().astype(int)])
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, clip)
    return clip


def ref_path(repo_file: str) -> Path:
    return REF_DIR / (repo_file.replace("/", "__").removesuffix(".mp4") + ".npz")


def sync() -> None:
    import torch

    torch.cuda.synchronize()


# ---- one configuration (own process) ---------------------------------------------------------------


def make_refs(files: list[str]) -> None:
    """Build the fp32 reference for every clip: true future latents, persistence step, baseline forecast."""
    from worldoptbench.models.vjepa2 import VJEPA2Latent

    model = VJEPA2Latent()
    n_ctx_steps = CONTEXT_FRAMES // model.config.tubelet_size
    per = model.tokens_per_step
    REF_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        clip = load_clip(f)
        rollout = model.generate(init_video=model.prepare(clip[:CONTEXT_FRAMES]), future_steps=n_ctx_steps)
        truth = model.encode_clip(model.prepare(clip))[n_ctx_steps * per : 2 * n_ctx_steps * per].numpy()
        np.savez(
            ref_path(f),
            truth=truth,
            persistence=rollout.metadata["context_last"],
            forecast=np.concatenate(rollout.frames),
        )
        print(f"   reference ready: {f}", flush=True)


def run_single(name: str, files: list[str], repeats: int, out_dir: Path) -> None:
    import torch

    from worldoptbench.metrics.latent import forecast_agreement, forecast_scores, split_steps
    from worldoptbench.models.vjepa2 import VJEPA2Latent
    from worldoptbench.stack import OptimizationStack

    modules, module_kwargs, model_kwargs = CONFIGS[name]
    if "torch_dtype" in model_kwargs:
        model_kwargs = {**model_kwargs, "torch_dtype": getattr(torch, model_kwargs["torch_dtype"])}
    row: dict = {"name": name, "error": None}
    try:
        model = VJEPA2Latent(**model_kwargs)
        stack = OptimizationStack(model, modules, module_kwargs=module_kwargs)
        if stack.skipped:  # a skipped module would silently turn this into a baseline row
            raise RuntimeError("; ".join(f"{s.name} skipped: {s.reason}" for s in stack.skipped))
        model = stack.apply()
        per = model.tokens_per_step
        n_future = CONTEXT_FRAMES // model.config.tubelet_size
        row["applied"] = [m.label for m in stack.modules]
        sync()
        row["held_gb"] = torch.cuda.memory_allocated() / 2**30

        clips = [model.prepare(load_clip(f)[:CONTEXT_FRAMES]) for f in files]
        t0 = time.perf_counter()  # warm-up: includes the compile for compiled configurations
        model.generate(init_video=clips[0], future_steps=n_future)
        sync()
        row["warmup_s"] = time.perf_counter() - t0
        model.generate(init_video=clips[0], future_steps=n_future)

        if name == "baseline":  # the constant that every configuration excludes from its timing
            raw = load_clip(files[0])[:CONTEXT_FRAMES]
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                model.prepare(raw)
                sync()
                times.append(time.perf_counter() - t0)
            row["preprocess_ms"] = st.median(times) * 1e3

        torch.cuda.reset_peak_memory_stats()
        per_clip: dict[str, list] = {k: [] for k in
            ("latency_ms", "cosine", "relative_l2", "skill", "drift_per_step", "agree_cosine", "agree_relative_l2")}
        for f, x in zip(files, clips):
            ref = np.load(ref_path(f))
            times = []
            for _ in range(repeats):
                sync()
                t0 = time.perf_counter()
                rollout = model.generate(init_video=x, future_steps=n_future)  # returns after the host copy: synced
                times.append(time.perf_counter() - t0)
            scores = forecast_scores(
                rollout.frames, split_steps(ref["truth"], per), ref["persistence"], steps_per_second=rollout.fps)
            agree = forecast_agreement(rollout.frames, split_steps(ref["forecast"], per))
            for key, value in (
                ("latency_ms", st.median(times) * 1e3), ("cosine", scores["cosine"]),
                ("relative_l2", scores["relative_l2"]), ("skill", scores["skill"]),
                ("drift_per_step", scores["drift_per_step"]), ("agree_cosine", agree["cosine"]),
                ("agree_relative_l2", agree["relative_l2"]),
            ):
                per_clip[key].append(value)
        row["clips"] = per_clip
        row["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
        row["n"] = len(files)
    except Exception as e:  # noqa: BLE001  (a failed configuration is reported in its row, not fatal)
        row["error"] = f"{type(e).__name__}: {e}"
    row_file(out_dir, name).write_text(json.dumps(row), encoding="utf-8")


def row_file(out_dir: Path, name: str) -> Path:
    return out_dir / f".row_{name.replace(' ', '_').replace('+', 'and')}.json"


def cell(clips: dict, key: str, decimals: int) -> str:
    mean, sem = mean_sem(clips[key])
    return f"{mean:.{decimals}f}+-{sem:.{decimals}f}"


# ---- the driver ------------------------------------------------------------------------------------


def mean_sem(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return float("nan"), 0.0
    return st.mean(vals), (st.stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="configuration names (baseline always runs first)")
    ap.add_argument("--clips-per-class", type=int, default=CLIPS_PER_CLASS)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "vjepa2")
    ap.add_argument("--single", help="(internal) run exactly this configuration in this process")
    ap.add_argument("--make-refs", action="store_true", help="(internal) build the fp32 references")
    args = ap.parse_args()

    files = select_clips(args.clips_per_class)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.make_refs:
        make_refs(files)
        return 0
    if args.single:
        run_single(args.single, files, args.repeats, args.out_dir)
        return 0

    names = ["baseline"] + [n for n in (args.only or CONFIGS) if n != "baseline"]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configuration(s) {unknown}; choose from {list(CONFIGS)}")
    me, py = str(Path(__file__).resolve()), sys.executable
    common = ["--clips-per-class", str(args.clips_per_class), "--repeats", str(args.repeats), "--out-dir", str(args.out_dir)]
    print("=== references (fp32) ...", flush=True)
    if subprocess.run([py, me, "--make-refs", *common], check=False).returncode:
        raise SystemExit("could not build the references")

    rows = []
    for name in names:
        print(f"=== {name} ...", flush=True)
        path = row_file(args.out_dir, name)
        path.unlink(missing_ok=True)
        proc = subprocess.run([py, me, "--single", name, *common], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", check=False)
        if not path.exists():
            rows.append({"name": name, "error": f"process failed ({proc.returncode}): {proc.stderr[-400:]}"})
        else:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        r = rows[-1]
        print("   ERROR: " + r["error"] if r["error"] else
              f"   {mean_sem(r['clips']['latency_ms'])[0]:.1f} ms  cos {mean_sem(r['clips']['cosine'])[0]:.4f}  "
              f"agree {mean_sem(r['clips']['agree_cosine'])[0]:.4f}  warmup {r['warmup_s']:.1f}s", flush=True)

    base = next(r for r in rows if r["name"] == "baseline")
    if base["error"]:
        raise SystemExit("the baseline failed, so there is nothing to compare against")
    for r in rows:
        if not r["error"]:
            r["clips"]["speedup"] = [b / x for b, x in zip(base["clips"]["latency_ms"], r["clips"]["latency_ms"])]
    (args.out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\n{'configuration':34s} {'ms':>8s} {'speedup':>14s} {'cos(true)':>16s} {'skill':>14s} "
          f"{'agree cos':>14s} {'agree relL2':>12s} {'peak GB':>8s} {'warmup s':>9s}")
    for r in rows:
        if r["error"]:
            print(f"{r['name']:34s} ERROR {r['error'][:110]}")
            continue
        c = r["clips"]
        print(f"{r['name']:34s} {mean_sem(c['latency_ms'])[0]:8.1f} {cell(c, 'speedup', 2):>14s} "
              f"{cell(c, 'cosine', 4):>16s} {cell(c, 'skill', 3):>14s} {cell(c, 'agree_cosine', 4):>14s} "
              f"{mean_sem(c['agree_relative_l2'])[0]:12.4f} {r['peak_gb']:8.2f} {r['warmup_s']:9.1f}")
    if "preprocess_ms" in base:
        print(f"\nCPU preprocessing (excluded from every timing above): {base['preprocess_ms']:.0f} ms per 32-frame context")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
