"""Comparison + Pareto plotting across benchmark result JSONs
(outline §7.2 step 5, §7.7).

Loading and summarizing results works with no extra dependencies.
`plot_pareto` needs the `plot` extra (`pip install -e ".[plot]"`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def load_results(path: Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text())


def drift_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Drift across scenarios. The runner fits one drift rate per prompt (its
    score-vs-horizon slope), so a run with N prompts has N drift estimates; one
    estimate is mostly noise (on a first real run the three scenarios gave -0.006,
    +0.001 and +0.005 per second), so what matters is the mean and its spread.

    `drift_detectable` is True only when the mean is more than two standard
    errors from zero and there are at least 3 scenarios — a rough guard against
    reading structure into noise, not a significance test.
    """
    per_prompt: dict[str, float] = {}
    for r in results:
        rate = (r.get("physics") or {}).get("drift_rate")
        if rate is not None:
            per_prompt[r["prompt_id"]] = rate
    rates = list(per_prompt.values())
    out: dict[str, Any] = {"n_scenarios": len(rates), "mean_drift_rate": None, "drift_rate_sem": None, "drift_detectable": None}
    if not rates:
        return out
    out["mean_drift_rate"] = mean(rates)
    if len(rates) >= 2:
        out["drift_rate_sem"] = stdev(rates) / math.sqrt(len(rates))
    if len(rates) >= 3 and out["drift_rate_sem"] is not None:
        out["drift_detectable"] = abs(out["mean_drift_rate"]) > 2 * out["drift_rate_sem"]
    return out


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates one run's results (one results.json) into a single
    comparison-table row.
    """
    if not results:
        return {"model": None, "architecture": None, "n_runs": 0}

    paes_values = [r["paes"] for r in results if r.get("paes") is not None]
    latencies = [r["speed"]["latency_seconds"] for r in results]
    speedups = [r["speed"]["speedup"] for r in results]

    return {
        "model": results[0]["model_name"],
        "architecture": results[0]["architecture"],
        # .get(): result files written before the field existed are baselines.
        "optimization": results[0].get("optimization", "baseline"),
        "n_runs": len(results),
        "mean_latency_seconds": mean(latencies) if latencies else None,
        "mean_speedup": mean(speedups) if speedups else None,
        "mean_paes": mean(paes_values) if paes_values else None,
        **drift_stats(results),
    }


def detect_hardware() -> str:
    """The GPU this process would run on (name and memory), or "CPU" without CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return f"{props.name} ({props.total_memory / 1024**3:.0f} GB)"
    except ImportError:
        pass
    return "CPU"


def format_profile(results: list[dict[str, Any]], hardware: str | None = None) -> str:
    """The per-run profile block of the outline (section 3.2): model, architecture, optimization, hardware, then
    speed, visual and physics lines and the PAES score. Takes one run's rows (RunResult dicts from a results JSON).
    Quantities a run did not measure are printed as n/a rather than left out."""
    if not results:
        return "(no results)"

    def values(get):
        return [v for r in results if (v := get(r)) is not None]

    def fmt(vals: list[float], spec: str = ".2f") -> str:
        return "n/a" if not vals else format(mean(vals), spec)

    speedups = values(lambda r: r["speed"].get("speedup"))
    per_second = values(lambda r: r["speed"]["latency_seconds"] / r["horizon"] if r.get("horizon") else None)
    vram = values(lambda r: r["speed"].get("vram_peak_gb"))
    reserved = values(lambda r: r["speed"].get("vram_reserved_gb"))
    drift = drift_stats(results)
    drift_text = "n/a"
    if drift["mean_drift_rate"] is not None:
        sem = drift["drift_rate_sem"]
        drift_text = f"{drift['mean_drift_rate']:+.4f}/s" + ("" if sem is None else f" +- {sem:.4f}")
        if drift["drift_detectable"] is False:
            drift_text += " (not distinguishable from zero)"
    style = values(lambda r: r["visual"].get("style_deviation"))
    dino = values(lambda r: (r["visual"].get("perceptual") or {}).get("dino_similarity"))
    extra = ""
    if dino:
        extra += f" | DINO similarity {fmt(dino, '.3f')}"
    if style:
        extra += f" | style deviation {fmt(style, '.3f')}"
    physics = []
    for r in results:
        p = r.get("physics") or {}
        score = p.get("sim_fidelity_score", p.get("score"))
        if score is not None:
            physics.append(score)

    lines = [
        f"Model:        {results[0]['model_name']}",
        f"Architecture: {results[0]['architecture']}",
        f"Optimization: {results[0].get('optimization', 'baseline')}",
        f"Hardware:     {hardware or detect_hardware()}",
        "",
        f"Speed:        {fmt(speedups)}x speedup | {fmt(per_second)} s/s generation | {fmt(vram)} GB VRAM peak"
        + (f" ({fmt(reserved)} GB held)" if reserved else ""),
        (f"Visual:       FVD n/a | PSNR {fmt(values(lambda r: r['visual'].get('psnr')), '.1f')} | "
        f"Temporal consistency {fmt(values(lambda r: r['visual'].get('temporal_consistency')), '.3f')}{extra}"),
        f"Physics:      {fmt(physics, '.3f')} | Drift onset: n/a (not computed) | Drift rate: {drift_text}",
        "",
        f"PAES Score:   {fmt(values(lambda r: r.get('paes')), '.3f')}",
    ]
    return "\n".join(lines)


def compare(result_paths: list[Path]) -> list[dict[str, Any]]:
    """One summary row per results JSON — for a quick side-by-side table
    across baseline / WorldCache / AdaCache / quantization runs etc.
    """
    return [summarize(load_results(p)) for p in result_paths]


def plot_pareto(
    result_paths: list[Path],
    labels: list[str] | None = None,
    output_path: Path | None = None,
):
    """Speed (x) vs. PAES (y) scatter across multiple result files — the
    Pareto-frontier plot the outline calls for in §7.2/§7.7. Points with no
    PAES (Phase 3 physics not wired up yet) are skipped.
    """
    import matplotlib.pyplot as plt

    labels = labels or [str(p) for p in result_paths]
    fig, ax = plt.subplots()

    for path, label in zip(result_paths, labels):
        results = load_results(path)
        scored = [r for r in results if r.get("paes") is not None]
        if not scored:
            continue
        speedups = [r["speed"]["speedup"] for r in scored]
        paes = [r["paes"] for r in scored]
        ax.scatter(speedups, paes, label=label)

    ax.set_xlabel("Speedup")
    ax.set_ylabel("PAES")
    ax.set_title("Speed vs. Physics-Aware Efficiency Score")
    ax.legend()

    if output_path:
        fig.savefig(output_path)
    return fig
