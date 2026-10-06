"""Measure diffusion optimizations on Wan2.1-T2V-1.3B.

    python scripts/sweep_wan.py                         # the default screening set (~25 min)
    python scripts/sweep_wan.py --only baseline fbc_0.1 # just these

Each configuration builds a fresh WanVideo, applies its modules, runs the benchmark, undoes
what it can, and frees the GPU. Failures are caught and reported per configuration (some
diffusers features may not work on the real pipeline), not allowed to end the sweep.

Quality is *agreement with the unoptimized video for the same prompt and seed* (the
reference is generated once and cached under results/wan_refs): PSNR / SSIM, and a "skill"
against repeating the reference's first frame. There is no physics score for Wan here, so
PAES's physics term means "fidelity to the baseline", not correctness. The baseline is run
twice, as a determinism check (its fidelity must be 1.0) and a timing-noise estimate.

Keep the GPU otherwise idle. Latency is read from the runner (generate() only; scenario
building and metrics are outside the timed region).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SET = ROOT / "worldoptbench" / "prompts" / "wan_set.json"

MODEL_KWARGS = {"height": 192, "width": 320, "num_inference_steps": 30, "guidance_scale": 5.0}

# name -> (modules in order, per-module kwargs)
CONFIGS: dict[str, tuple[list[str], dict[str, dict]]] = {
    "baseline": ([], {}),
    "baseline (repeat)": ([], {}),
    "fbc_0.05": (["first_block_cache"], {"first_block_cache": {"threshold": 0.05}}),
    "fbc_0.1": (["first_block_cache"], {"first_block_cache": {"threshold": 0.1}}),
    "fbc_0.2": (["first_block_cache"], {"first_block_cache": {"threshold": 0.2}}),
    "pab_2": (["pab"], {"pab": {"block_skip_range": 2}}),
    "pab_3": (["pab"], {"pab": {"block_skip_range": 3}}),
    "taylorseer_3": (["taylorseer"], {"taylorseer": {"cache_interval": 3}}),
    "taylorseer_5": (["taylorseer"], {"taylorseer": {"cache_interval": 5}}),
    "layer_skip_2": (["layer_skip"], {"layer_skip": {"num_blocks": 2}}),
    "layer_skip_4": (["layer_skip"], {"layer_skip": {"num_blocks": 4}}),
    "attn_native": (["attention_backend"], {"attention_backend": {"backend": "native"}}),
    "attn_flex": (["attention_backend"], {"attention_backend": {"backend": "flex"}}),
    "attn_cudnn": (["attention_backend"], {"attention_backend": {"backend": "_native_cudnn"}}),
    "attn_efficient": (["attention_backend"], {"attention_backend": {"backend": "_native_efficient"}}),
    "attn_flash_sdpa": (["attention_backend"], {"attention_backend": {"backend": "_native_flash"}}),
    "fp8_storage": (["layerwise_casting"], {}),
    "cfg_trunc_0.6": (["cfg_truncation"], {"cfg_truncation": {"after_fraction": 0.6}}),
    "cfg_trunc_0.4": (["cfg_truncation"], {"cfg_truncation": {"after_fraction": 0.4}}),
    "cfg0.6+pab_2": (
        ["pab", "cfg_truncation"], {"pab": {"block_skip_range": 2}, "cfg_truncation": {"after_fraction": 0.6}}
    ),
    "cfg0.6+pab_3": (
        ["pab", "cfg_truncation"], {"pab": {"block_skip_range": 3}, "cfg_truncation": {"after_fraction": 0.6}}
    ),
    "cfg0.4+pab_2": (
        ["pab", "cfg_truncation"], {"pab": {"block_skip_range": 2}, "cfg_truncation": {"after_fraction": 0.4}}
    ),
    "cfg0.6+fbc_0.05": (
        ["first_block_cache", "cfg_truncation"],
        {"first_block_cache": {"threshold": 0.05}, "cfg_truncation": {"after_fraction": 0.6}},
    ),
    "wc_0.02": (["worldcache"], {"worldcache": {"tau0": 0.02}}),
    "wc_0.04": (["worldcache"], {"worldcache": {"tau0": 0.04}}),
    "wc_0.08": (["worldcache"], {"worldcache": {"tau0": 0.08}}),
    "wc_0.04_nointerp": (["worldcache"], {"worldcache": {"tau0": 0.04, "interpolate": False}}),
    "wc_0.04_nosched": (["worldcache"], {"worldcache": {"tau0": 0.04, "schedule": False}}),
    "wc_0.04_nosal": (["worldcache"], {"worldcache": {"tau0": 0.04, "saliency": False}}),
    "wc_0.04_nomotion": (["worldcache"], {"worldcache": {"tau0": 0.04, "motion_adaptive": False}}),
    "wc_0.04_warp": (["worldcache"], {"worldcache": {"tau0": 0.04, "warp": True}}),
    "ada_slow30": (["adacache"], {"adacache": {"preset": "slow30"}}),
    "ada_slow30_x0.5": (["adacache"], {"adacache": {"preset": "slow30", "distance_scale": 0.5}}),
    "ada_slow30_x0.25": (["adacache"], {"adacache": {"preset": "slow30", "distance_scale": 0.25}}),
    "ada_slow30_x2": (["adacache"], {"adacache": {"preset": "slow30", "distance_scale": 2.0}}),
    "ada_slow30_x4": (["adacache"], {"adacache": {"preset": "slow30", "distance_scale": 4.0}}),
    "ada_fast30_x4": (["adacache"], {"adacache": {"preset": "fast30", "distance_scale": 4.0}}),
    "ada_fast30_guard0.05": (["adacache"], {"adacache": {"preset": "fast30", "guard": 0.05}}),
    "ada_fast30_guard0.1": (["adacache"], {"adacache": {"preset": "fast30", "guard": 0.1}}),
    "ada_fast30_guard0.15": (["adacache"], {"adacache": {"preset": "fast30", "guard": 0.15}}),
    "ada_fast30_guard0.25": (["adacache"], {"adacache": {"preset": "fast30", "guard": 0.25}}),
    "ada_slow30_guard0.15": (["adacache"], {"adacache": {"preset": "slow30", "guard": 0.15}}),
    "cfg0.6+ada_fast30_guard0.15": (
        ["adacache", "cfg_truncation"], {"adacache": {"preset": "fast30", "guard": 0.15}, "cfg_truncation": {"after_fraction": 0.6}}
    ),
    "ada_fast30": (["adacache"], {"adacache": {"preset": "fast30"}}),
    "ada_fast30_x2": (["adacache"], {"adacache": {"preset": "fast30", "distance_scale": 2.0}}),
    "ada_fast30_moreg": (["adacache"], {"adacache": {"preset": "fast30", "moreg": True}}),
    "ada_fast30_x2_moreg": (["adacache"], {"adacache": {"preset": "fast30", "distance_scale": 2.0, "moreg": True}}),
    "wc_0.08_prevstep": (["worldcache"], {"worldcache": {"tau0": 0.08, "drift_reference": "previous_step"}}),
    "cfg0.6+wc_0.04": (
        ["worldcache", "cfg_truncation"], {"worldcache": {"tau0": 0.04}, "cfg_truncation": {"after_fraction": 0.6}}
    ),
    # Controls, not optimizations: perturb the first step's latents by this much (std, vs unit-variance
    # noise) to measure how far a numerically-equivalent change moves the metrics (the noise floor).
    "perturb_1e-4": ([], {}),
    "perturb_1e-2": ([], {}),
    "sched_dpmpp_s20": (["fewer_steps", "scheduler"], {"fewer_steps": {"steps": 20}, "scheduler": {"kind": "dpmpp"}}),
    "sched_euler_s20": (["fewer_steps", "scheduler"], {"fewer_steps": {"steps": 20}, "scheduler": {"kind": "euler"}}),
    "sched_unipc_o3_s20": (["fewer_steps", "scheduler"], {"fewer_steps": {"steps": 20}, "scheduler": {"kind": "unipc", "solver_order": 3}}),
    "sched_unipc_shift5_s20": (["fewer_steps", "scheduler"], {"fewer_steps": {"steps": 20}, "scheduler": {"kind": "unipc", "flow_shift": 5.0}}),
    "sched_unipc_shift1.5_s20": (["fewer_steps", "scheduler"], {"fewer_steps": {"steps": 20}, "scheduler": {"kind": "unipc", "flow_shift": 1.5}}),
    "uncond_reuse_2": (["uncond_reuse"], {"uncond_reuse": {"period": 2}}),
    "uncond_reuse_3": (["uncond_reuse"], {"uncond_reuse": {"period": 3}}),
    "kv_cache": (["cross_attn_kv_cache"], {}),
    "vae_tiling": (["vae"], {"vae": {"tiling": True}}),
    "vae_bf16": (["vae"], {"vae": {"tiling": False, "dtype": "bfloat16"}}),
    "kv+uncond2+cfg0.6": (["cross_attn_kv_cache", "uncond_reuse", "cfg_truncation"],
                          {"uncond_reuse": {"period": 2}, "cfg_truncation": {"after_fraction": 0.6}}),
    # MagCache needs per-step magnitude ratios measured on this exact setup; run_one calibrates once and caches them
    "mag_0.06": (["magcache"], {"magcache": {"threshold": 0.06}}),
    "mag_0.12": (["magcache"], {"magcache": {"threshold": 0.12}}),
    "mag_0.24": (["magcache"], {"magcache": {"threshold": 0.24}}),
    "mag_0.5": (["magcache"], {"magcache": {"threshold": 0.5}}),
    "mag_1.0": (["magcache"], {"magcache": {"threshold": 1.0}}),
    "mag_0.24_skip5": (["magcache"], {"magcache": {"threshold": 0.24, "max_skip_steps": 5}}),
    "mag_0.5_skip5": (["magcache"], {"magcache": {"threshold": 0.5, "max_skip_steps": 5}}),
    "mag_0.24_ret0.1": (["magcache"], {"magcache": {"threshold": 0.24, "retention_ratio": 0.1}}),
    "mag_0.24+cfg0.6": (["magcache", "cfg_truncation"], {"magcache": {"threshold": 0.24}, "cfg_truncation": {"after_fraction": 0.6}}),
    "mag_0.24+cfg0.6+kv+vae": (["magcache", "cfg_truncation", "cross_attn_kv_cache", "vae"],
                               {"magcache": {"threshold": 0.24}, "cfg_truncation": {"after_fraction": 0.6}, "vae": {"tiling": False, "dtype": "bfloat16"}}),
    "mag_0.5+cfg0.6+kv+vae": (["magcache", "cfg_truncation", "cross_attn_kv_cache", "vae"],
                              {"magcache": {"threshold": 0.5}, "cfg_truncation": {"after_fraction": 0.6}, "vae": {"tiling": False, "dtype": "bfloat16"}}),
    "tf32": (["tf32"], {}),
    "cudnn_bench": (["cudnn_benchmark"], {}),
    "sched_unipc_shift8_s20": (["fewer_steps", "scheduler"], {"fewer_steps": {"steps": 20}, "scheduler": {"kind": "unipc", "flow_shift": 8.0}}),
    "sched_unipc_shift5_s15": (["fewer_steps", "scheduler"], {"fewer_steps": {"steps": 15}, "scheduler": {"kind": "unipc", "flow_shift": 5.0}}),
    "s20shift5+cfg0.6": (["fewer_steps", "scheduler", "cfg_truncation"],
                         {"fewer_steps": {"steps": 20}, "scheduler": {"kind": "unipc", "flow_shift": 5.0}, "cfg_truncation": {"after_fraction": 0.6}}),
    "s20shift5+cfg0.4+pab": (["fewer_steps", "scheduler", "pab", "cfg_truncation"],
                             {"fewer_steps": {"steps": 20}, "scheduler": {"kind": "unipc", "flow_shift": 5.0},
                              "pab": {"block_skip_range": 2}, "cfg_truncation": {"after_fraction": 0.4}}),
    "lossless_kv+vae_bf16": (["cross_attn_kv_cache", "vae"], {"vae": {"tiling": False, "dtype": "bfloat16"}}),
    "kv+vae_bf16+cfg0.4+pab": (["cross_attn_kv_cache", "vae", "pab", "cfg_truncation"],
                               {"vae": {"tiling": False, "dtype": "bfloat16"}, "pab": {"block_skip_range": 2}, "cfg_truncation": {"after_fraction": 0.4}}),
    "steps_20": (["fewer_steps"], {"fewer_steps": {"steps": 20}}),
    "steps_15": (["fewer_steps"], {"fewer_steps": {"steps": 15}}),
}


def _perturbation(scale: float):
    """A step callback adding N(0, scale^2) to the first step's latents (fixed noise, so it is the same
    perturbation for every clip)."""

    def callback(pipe, step, timestep, callback_kwargs):
        if step == 0:
            import torch

            latents = callback_kwargs["latents"]
            generator = torch.Generator(device=latents.device).manual_seed(123)
            noise = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
            callback_kwargs["latents"] = latents + scale * noise
        return callback_kwargs

    return callback


def _mag_ratios(model) -> list[float]:
    """MagCache's per-step magnitude ratios for this model, size and step count: measured once (a full generation in
    calibration mode) and cached in results/wan_mag_ratios.json, so each configuration's process does not repeat it."""
    from worldoptbench.optimizations.diffusion_more import calibrate_mag_ratios

    path = ROOT / "results" / "wan_mag_ratios.json"
    key = f"{model.height}x{model.width}_s{model.num_inference_steps}_{type(model.pipeline.scheduler).__name__}"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if key not in cache:
        cache[key] = calibrate_mag_ratios(model, lambda: model.generate(prompt=model.prompts[0], horizon=2, seed=0))
        path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache[key]


def _baseline_latency(out_dir: Path) -> dict[str, float]:
    path = out_dir / "baseline.json"
    if not path.exists():
        return {}
    from worldoptbench.runner import latency_key

    return {latency_key(r["prompt_id"], r["horizon"]): r["speed"]["latency_seconds"] for r in json.loads(path.read_text(encoding="utf-8"))}


def run_one(name: str, standard_set: Path, out_dir: Path) -> dict:
    """Runs a single configuration in THIS process and returns its summary row."""
    import torch

    from worldoptbench.models.wan_video import WanReferenceScenarios, WanVideo
    from worldoptbench.runner import run_benchmark
    from worldoptbench.stack import OptimizationStack

    modules, module_kwargs = CONFIGS[name]
    scenarios = WanReferenceScenarios(ROOT / "results" / "wan_refs", model_kwargs=MODEL_KWARGS)
    stem = name.replace(" ", "_").replace("(", "").replace(")", "")
    torch.cuda.reset_peak_memory_stats()
    model = WanVideo(**MODEL_KWARGS)
    if "magcache" in modules:
        module_kwargs = {**module_kwargs, "magcache": {**module_kwargs["magcache"], "mag_ratios": _mag_ratios(model)}}
    stack = OptimizationStack(model, modules, module_kwargs=module_kwargs)
    if name.startswith("perturb_"):
        model.step_callbacks.append(_perturbation(float(name.split("_", 1)[1])))
    if stack.skipped:
        print("   skipped:", "; ".join(f"{s.name}: {s.reason}" for s in stack.skipped), flush=True)
    from worldoptbench.metrics.perceptual import PerceptualScorer

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
        "skill": per_clip(lambda r: r.physics["sim_fidelity_score"] if r.physics else None),
        "style": per_clip(lambda r: r.visual.get("style_deviation")),
        "dino_sim": per_clip(lambda r: (r.visual.get("perceptual") or {}).get("dino_similarity")),
        "dino_consistency": per_clip(lambda r: (r.visual.get("perceptual") or {}).get("dino_consistency")),
        "clip_delta": per_clip(lambda r: (r.visual.get("perceptual") or {}).get("clip_delta")),
    }
    return {
        "name": name, "applied": [m.name for m in stack.modules], "skipped": [s.name for s in stack.skipped],
        "n": len(results), "clips": clips,
        "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
        "error": None,
    }


def mean_sem(values: list[float]) -> tuple[float, float]:
    """Mean and standard error of the mean over clips (inf PSNR, from a clip identical to the
    baseline, is capped at 60 dB so one perfect clip doesn't make the mean infinite)."""
    values = [min(v, 60.0) for v in values]
    mean = st.mean(values)
    return mean, (st.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="configuration names to run (baseline always runs first)")
    ap.add_argument("--standard-set", type=Path, default=SET)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "wan")
    ap.add_argument("--single", help="(internal) run exactly this configuration in this process")
    ap.add_argument("--skip-baseline", action="store_true", help="reuse an existing baseline.json instead of rerunning it")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.single:
        row = run_one(args.single, args.standard_set, args.out_dir)
        (args.out_dir / f".row_{args.single.replace(' ', '_')}.json").write_text(json.dumps(row), encoding="utf-8")
        return 0

    # Each configuration runs in its own process. An in-process sweep lets global state leak between
    # configurations: a failed `_native_flash` attention backend left PyTorch's SDPA settings broken
    # for every later configuration ("No available kernel"), contaminating five unrelated results.
    names = ([] if args.skip_baseline else ["baseline"]) + [
        n for n in CONFIGS if n != "baseline" and (not args.only or n in args.only)
    ]
    rows: list[dict] = []
    for name in names:
        print(f"=== {name} ...", flush=True)
        row_file = args.out_dir / f".row_{name.replace(' ', '_')}.json"
        row_file.unlink(missing_ok=True)
        cmd = [sys.executable, str(Path(__file__)), "--single", name, "--standard-set", str(args.standard_set), "--out-dir", str(args.out_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and row_file.exists():
            rows.append(json.loads(row_file.read_text(encoding="utf-8")))
        else:
            last = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()][-1:]
            rows.append({"name": name, "error": (last[0] if last else f"exit code {proc.returncode}")[:160]})

    print()
    print("mean +- standard error over clips (n per row below); PSNR capped at 60 dB")
    print(f"{'configuration':22s} {'n':>3s} {'latency s':>13s} {'speedup':>13s} {'PSNR':>12s} {'SSIM':>13s} {'skill':>13s} {'style dev':>13s} {'dino sim':>13s} {'clip delta':>13s} {'peak GB':>8s}  notes")
    for r in rows:
        if r["error"]:
            print(f"{r['name']:22s} FAILED: {r['error']}")
            continue
        c = r["clips"]
        cell = lambda key, w=5, d=2: "{:{w}.{d}f} +-{:.{d}f}".format(*mean_sem(c[key]), w=w, d=d)
        notes = ("skipped: " + ",".join(r["skipped"])) if r["skipped"] else ""
        print(f"{r['name']:22s} {r['n']:3d} {cell('latency', 5, 1):>13s} {cell('speedup', 5, 2):>13s} {cell('psnr', 5, 1):>12s} {cell('ssim', 5, 3):>13s} {cell('skill', 5, 3):>13s} {cell('style', 5, 3):>13s} {cell('dino_sim', 5, 3):>13s} {cell('clip_delta', 5, 3):>13s} {r['peak_gb']:8.2f}  {notes}")
    (args.out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
