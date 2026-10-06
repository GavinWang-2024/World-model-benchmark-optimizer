"""A rule-based default stack, for when you don't want to run `autotune`.

This encodes what was *measured* on the two models we have (a trained DreamerV3
walker, see DREAMER_SETUP.md, and Wan2.1-1.3B, see DESIGN_DIFFUSION.md, both on an
RTX 5070 Ti laptop GPU): the optimizations that helped, in order of preference, falling
back to whatever the model and machine support. For Dreamer the tiers keep the output
(essentially) unchanged; for diffusion no speedup is free, so its tiers are a *policy*:
the knee of the measured speed/fidelity curve, not a lossless setting. It is
a starting point, not a guarantee — a different model or GPU can rank things
differently (quantization and mixed precision, for instance, hurt here but are
standard wins on large models), which is exactly what `autotune` exists to check.
Its first end-to-end run on held-out walking episodes independently chose the top
tier below (`cuda_graphs` + `tensorrt` at fp16).

Only `measured` modules are ever recommended; an experimental one is never put in a
default just because it exists.
"""

from __future__ import annotations

from typing import Any

from worldoptbench.optimizations.base import available_modules, get_module_class

# Tiers, best first. Each is (modules in application order, per-module constructor kwargs); a tier
# is used only if *every* module in it applies to the model here.
# Only with `aggressive=True`: MagCache, which needs per-setup magnitude ratios (the model supplies them for a setup it was
# calibrated on, e.g. WanVideo at 192x320 / 30 steps; otherwise the tier is skipped). Wan2.1-1.3B, 32 clips: 2.54x, DINO similarity
# 0.856 (the noise control is 0.876; 3 of 32 clips below the floor), prompt match unchanged, but a small measurable style shift
# (pixel style deviation 0.12 against the control's 0.08) and a mild softening of fine detail visible on some clips.
_FAST_TIERS: tuple[tuple[tuple[str, ...], dict[str, dict[str, Any]]], ...] = (
    (
        ("magcache", "cfg_truncation", "cross_attn_kv_cache", "vae"),
        {
            "magcache": {"threshold": 0.24},
            "cfg_truncation": {"after_fraction": 0.6},
            "vae": {"tiling": False, "dtype": "bfloat16"},
        },
    ),
)

_TIERS: tuple[tuple[tuple[str, ...], dict[str, dict[str, Any]]], ...] = (
    # Diffusion (Wan2.1-1.3B, 32 clips; PSNR = agreement with the unoptimized video). First, because the
    # Dreamer-era fallbacks below would otherwise match a diffusion model too (no_dist_validation applies to
    # anything); a Dreamer model never matches these, since the modules refuse its architecture.
    (  # 1.54x, indistinguishable from the unoptimized video on every measure used (DINO 0.93, 0 of 32 clips below the noise floor)
        ("pab", "cfg_truncation", "cross_attn_kv_cache", "vae"),
        {
            "pab": {"block_skip_range": 2},
            "cfg_truncation": {"after_fraction": 0.4},
            "vae": {"tiling": False, "dtype": "bfloat16"},
        },
    ),
    (  # the same without the bf16 VAE and KV cache: 1.43x at PSNR 26.3 (used where the pipeline has no vae)
        ("pab", "cfg_truncation"),
        {"pab": {"block_skip_range": 2}, "cfg_truncation": {"after_fraction": 0.4}},
    ),
    (("cfg_truncation",), {"cfg_truncation": {"after_fraction": 0.6}}),  # 1.21x at PSNR 30.8: the gentler option
    # 9.9 ms at the sweep's mix of horizons, the fastest measured; physics matches fp32 (autotune chose it too)
    (("tensorrt", "cuda_graphs"), {"tensorrt": {"precision": "fp16"}}),
    (("gumbel_sampling", "cuda_graphs"), {}),  # 17.6 ms: same without the TensorRT dependency
    (("cuda_graphs",), {}),  # 21.7 ms
    (("no_dist_validation",), {}),  # ~1.4x on the eager path, where graphs aren't available
)


def recommended_config(model: Any, aggressive: bool = False) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """The best measured stack whose modules all apply to `model` on this machine,
    as (module names, per-module kwargs) — pass both to `OptimizationStack`. Returns
    ([], {}) (run unoptimized) if no tier applies.

    `aggressive=True` tries the faster, slightly lossier diffusion tier (MagCache, about 2.5x on Wan with a small measurable
    shift) before the default ones; it applies only to a model that has MagCache ratios for its setup and otherwise
    falls back to the default tiers.
    """
    registered = set(available_modules())
    for names, kwargs in (*(_FAST_TIERS if aggressive else ()), *_TIERS):
        if not all(name in registered for name in names):
            continue
        modules = [get_module_class(name)() for name in names]
        if all(m.maturity == "measured" and m.incompatibility(model) is None for m in modules):
            return list(names), {k: dict(v) for k, v in kwargs.items()}
    return [], {}


def recommended_stack(model: Any) -> list[str]:
    """Just the module names from `recommended_config`. Note that the top tier needs its
    kwargs (fp16) to be the measured configuration; prefer `recommended_config`.
    """
    return recommended_config(model)[0]
