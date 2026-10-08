"""Measure diffusion optimizations on Cosmos-Predict2.5-2B (the Cosmos counterpart of scripts/sweep_wan.py).

    python scripts/sweep_cosmos.py                          # the screening set
    python scripts/sweep_cosmos.py --only baseline cfg_trunc_0.6

Same method as the Wan sweep: every configuration runs in its own process (global state such as attention-backend settings
must not leak between configurations), failures are reported per configuration, and quality is agreement with the
unoptimized video for the same prompt and seed (references are built once, in their own process BEFORE any model under
test is loaded, and cached under results/cosmos_refs). Two controls, `perturb_1e-4` and `perturb_1e-2`, set the noise
floor: a numerically equivalent change moves the metrics by that much, so only effects beyond it count.

Scale: ~110 s per 17-frame clip at 480x832 (36 steps) on a 12 GB laptop GPU, so a 4-clip configuration takes ~10 minutes.
Four clips is a screen, not a result: confirm anything promising on more prompts. The safety guardrail's ~0.2 s per clip is
inside every measured latency (a constant, so speedups are slightly understated). Keep the GPU otherwise idle.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_wan import _baseline_latency, _perturbation

SET = ROOT / "worldoptbench" / "prompts" / "cosmos_set.json"
REFS = ROOT / "results" / "cosmos_refs"
MODEL_KWARGS = {"height": 480, "width": 832, "num_inference_steps": 36, "guidance_scale": 7.0}

_BF16_TILED = {"vae": {"tiling": True, "dtype": "bfloat16"}}

# name -> (modules in order, per-module kwargs)
CONFIGS: dict[str, tuple[list[str], dict[str, dict]]] = {
    "baseline": ([], {}),
    "baseline (repeat)": ([], {}),
    # controls: how far a numerically equivalent change moves the metrics
    "perturb_1e-4": ([], {}),
    "perturb_1e-2": ([], {}),
    "cfg_trunc_0.6": (["cfg_truncation"], {"cfg_truncation": {"after_fraction": 0.6}}),
    "cfg_trunc_0.4": (["cfg_truncation"], {"cfg_truncation": {"after_fraction": 0.4}}),
    "steps_24": (["fewer_steps"], {"fewer_steps": {"steps": 24}}),
    "steps_18": (["fewer_steps"], {"fewer_steps": {"steps": 18}}),
    "attn_native": (["attention_backend"], {"attention_backend": {"backend": "native"}}),
    "attn_cudnn": (["attention_backend"], {"attention_backend": {"backend": "_native_cudnn"}}),
    "attn_efficient": (["attention_backend"], {"attention_backend": {"backend": "_native_efficient"}}),
    "fp8_storage": (["layerwise_casting"], {}),
    "tf32": (["tf32"], {}),
    "cudnn_bench": (["cudnn_benchmark"], {}),
    "vae_bf16": (["vae"], _BF16_TILED),
    "ada_slow30_x4": (["adacache"], {"adacache": {"preset": "slow30", "distance_scale": 4.0}}),
    "ada_fast30_guard0.15": (["adacache"], {"adacache": {"preset": "fast30", "guard": 0.15}}),
    "steps_24+cfg0.6": (["fewer_steps", "cfg_truncation"], {"fewer_steps": {"steps": 24}, "cfg_truncation": {"after_fraction": 0.6}}),
}


# The block-level caches. They work on Cosmos only because `models.cosmos_predict.make_cacheable` gives its transformer
# the cache API and per-guidance-pass cache contexts that diffusers' pipeline would otherwise supply. A second pass, run after
# the screening set above (`--only` these names with `--skip-baseline`).
CACHE_CONFIGS: dict[str, tuple[list[str], dict[str, dict]]] = {
    "fbc_0.05": (["first_block_cache"], {"first_block_cache": {"threshold": 0.05}}),
    "fbc_0.1": (["first_block_cache"], {"first_block_cache": {"threshold": 0.1}}),
    "fbc_0.2": (["first_block_cache"], {"first_block_cache": {"threshold": 0.2}}),
    "pab_2": (["pab"], {"pab": {"block_skip_range": 2}}),
    "pab_3": (["pab"], {"pab": {"block_skip_range": 3}}),
    "taylorseer_3": (["taylorseer"], {"taylorseer": {"cache_interval": 3}}),
    "taylorseer_5": (["taylorseer"], {"taylorseer": {"cache_interval": 5}}),
    "layer_skip_2": (["layer_skip"], {"layer_skip": {"num_blocks": 2}}),
    "wc_0.02": (["worldcache"], {"worldcache": {"tau0": 0.02}}),
    "wc_0.04": (["worldcache"], {"worldcache": {"tau0": 0.04}}),
    "wc_0.08": (["worldcache"], {"worldcache": {"tau0": 0.08}}),
    "kv_cache": (["cross_attn_kv_cache"], {}),
    "uncond_reuse_2": (["uncond_reuse"], {"uncond_reuse": {"period": 2}}),
    "cfg0.6+pab_2": (["pab", "cfg_truncation"], {"pab": {"block_skip_range": 2}, "cfg_truncation": {"after_fraction": 0.6}}),
    "cfg0.6+wc_0.04": (["worldcache", "cfg_truncation"], {"worldcache": {"tau0": 0.04}, "cfg_truncation": {"after_fraction": 0.6}}),
    "cfg0.6+fbc_0.05": (["first_block_cache", "cfg_truncation"], {"first_block_cache": {"threshold": 0.05}, "cfg_truncation": {"after_fraction": 0.6}}),
}
CONFIGS.update(CACHE_CONFIGS)


def mean_sem(values: list[float], cap: float | None = None) -> tuple[float, float]:
    """Mean and standard error over clips. `cap` clips each value first: PSNR needs it (a clip identical to the reference is
    infinite dB); latencies must not be capped (`sweep_wan.mean_sem` caps everything at 60, and Cosmos clips take longer)."""
    values = [min(v, cap) for v in values] if cap is not None else list(values)
    return st.mean(values), (st.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0)


def make_refs(standard_set: Path) -> None:
    """Builds (or loads) every reference video in THIS process, with no model under test loaded."""
    from worldoptbench.models.cosmos_predict import CosmosReferenceScenarios

    entries = json.loads(standard_set.read_text(encoding="utf-8"))
    scenarios = CosmosReferenceScenarios(REFS, model_kwargs=MODEL_KWARGS)
    for entry in entries["prompts"]:
        for horizon in entries["horizons_seconds"]:
            scenarios(entry, horizon)
            print(f"   reference ready: {entry['id']} @ {horizon:g}s", flush=True)


def run_one(name: str, standard_set: Path, out_dir: Path) -> dict:
    """Runs a single configuration in THIS process and returns its summary row."""
    import torch

    from worldoptbench.metrics.perceptual import PerceptualScorer
    from worldoptbench.models.cosmos_predict import CosmosPredict, CosmosReferenceScenarios
    from worldoptbench.runner import run_benchmark
    from worldoptbench.stack import OptimizationStack

    modules, module_kwargs = CONFIGS[name]
    scenarios = CosmosReferenceScenarios(REFS, model_kwargs=MODEL_KWARGS)
    stem = name.replace(" ", "_").replace("(", "").replace(")", "")
    # Every reference is cached, so this builds no second model; if one is missing, fail rather than load two models.
    entries = json.loads(standard_set.read_text(encoding="utf-8"))
    for entry in entries["prompts"]:
        for horizon in entries["horizons_seconds"]:
            if not scenarios._path(entry["prompt"], int(entry.get("seed", 0)), horizon).exists():
                raise SystemExit("references are missing; run without --single so they are built first")
    torch.cuda.reset_peak_memory_stats()
    model = CosmosPredict(**MODEL_KWARGS)
    stack = OptimizationStack(model, modules, module_kwargs=module_kwargs)
    if name.startswith("perturb_"):
        model.step_callbacks.append(_perturbation(float(name.split("_", 1)[1])))
    if stack.skipped:
        print("   skipped:", "; ".join(f"{s.name}: {s.reason}" for s in stack.skipped), flush=True)
    results = run_benchmark(
        stack.apply(),
        perceptual_scorer=PerceptualScorer(keep_descriptor=True),
        standard_set_path=standard_set,
        scenario_fn=scenarios,
        baseline_latencies=_baseline_latency(out_dir) or None,
        optimization=name,
        output_path=out_dir / f"{stem}.json",
    )

    def per_clip(get):
        return [get(r) for r in results if get(r) is not None]

    clips = {
        "latency": per_clip(lambda r: r.speed["latency_seconds"]),
        "speedup": per_clip(lambda r: r.speed["speedup"]),
        "psnr": per_clip(lambda r: r.visual.get("psnr")),
        "ssim": per_clip(lambda r: r.visual.get("ssim")),
        "style": per_clip(lambda r: r.visual.get("style_deviation")),
        "dino_sim": per_clip(lambda r: (r.visual.get("perceptual") or {}).get("dino_similarity")),
        "clip_delta": per_clip(lambda r: (r.visual.get("perceptual") or {}).get("clip_delta")),
    }
    return {
        "name": name, "applied": [m.name for m in stack.modules], "skipped": [s.name for s in stack.skipped],
        "n": len(results), "clips": clips, "peak_gb": torch.cuda.max_memory_allocated() / 2**30, "error": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="configuration names to run (baseline always runs first)")
    ap.add_argument("--standard-set", type=Path, default=SET)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "cosmos")
    ap.add_argument("--single", help="(internal) run exactly this configuration in this process")
    ap.add_argument("--make-refs", action="store_true", help="(internal) build the reference videos and exit")
    ap.add_argument("--skip-baseline", action="store_true", help="reuse an existing baseline.json instead of rerunning it")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.make_refs:
        make_refs(args.standard_set)
        return 0
    if args.single:
        row = run_one(args.single, args.standard_set, args.out_dir)
        (args.out_dir / f".row_{args.single.replace(' ', '_')}.json").write_text(json.dumps(row), encoding="utf-8")
        return 0

    py, me, common = sys.executable, str(Path(__file__)), ["--standard-set", str(args.standard_set), "--out-dir", str(args.out_dir)]
    print("=== references ...", flush=True)
    refs = subprocess.run([py, me, "--make-refs", *common], check=False)
    if refs.returncode != 0:
        raise SystemExit("building the reference videos failed")

    names = ([] if args.skip_baseline else ["baseline"]) + [
        n for n in CONFIGS if n != "baseline" and (n in args.only if args.only else n not in CACHE_CONFIGS)
    ]
    # Resuming: keep the rows of configurations finished by an earlier run (their `.row_*.json` files) that this run will not redo.
    rows: list[dict] = []
    for name in CONFIGS:
        earlier = args.out_dir / f".row_{name.replace(' ', '_')}.json"
        if name not in names and earlier.exists():
            rows.append(json.loads(earlier.read_text(encoding="utf-8")))
    for name in names:
        print(f"=== {name} ...", flush=True)
        row_file = args.out_dir / f".row_{name.replace(' ', '_')}.json"
        row_file.unlink(missing_ok=True)
        proc = subprocess.run([py, me, "--single", name, *common], capture_output=True, text=True, check=False)
        if proc.returncode == 0 and row_file.exists():
            rows.append(json.loads(row_file.read_text(encoding="utf-8")))
        else:
            last = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()][-1:]
            rows.append({"name": name, "error": (last[0] if last else f"exit code {proc.returncode}")[:200]})
        r = rows[-1]
        if r["error"]:
            print(f"   FAILED: {r['error']}", flush=True)
        else:
            c = r["clips"]
            print(f"   latency {mean_sem(c['latency'])[0]:.1f}s speedup {mean_sem(c['speedup'])[0]:.2f}x "
                  f"psnr {mean_sem(c['psnr'], cap=60.0)[0]:.1f} dino {mean_sem(c['dino_sim'])[0]:.3f}", flush=True)
        (args.out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print()
    print("mean +- standard error over clips; PSNR capped at 60 dB")
    print(f"{'configuration':24s} {'n':>3s} {'latency s':>13s} {'speedup':>13s} {'PSNR':>12s} {'SSIM':>13s} {'style dev':>13s} {'dino sim':>13s} {'clip delta':>13s} {'peak GB':>8s}  notes")
    for r in rows:
        if r["error"]:
            print(f"{r['name']:24s} FAILED: {r['error']}")
            continue
        c = r["clips"]

        def cell(key, w=5, d=2, clips=c):
            return "{:{w}.{d}f} +-{:.{d}f}".format(*mean_sem(clips[key], cap=60.0 if key == "psnr" else None), w=w, d=d)

        notes = ("skipped: " + ",".join(r["skipped"])) if r["skipped"] else ""
        print(f"{r['name']:24s} {r['n']:3d} {cell('latency', 5, 1):>13s} {cell('speedup', 5, 2):>13s} {cell('psnr', 5, 1):>12s} {cell('ssim', 5, 3):>13s} {cell('style', 5, 3):>13s} {cell('dino_sim', 5, 3):>13s} {cell('clip_delta', 5, 3):>13s} {r['peak_gb']:8.2f}  {notes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
