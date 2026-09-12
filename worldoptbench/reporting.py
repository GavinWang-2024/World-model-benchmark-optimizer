"""Comparison + Pareto plotting across benchmark result JSONs
(outline §7.2 step 5, §7.7).

Loading and summarizing results works with no extra dependencies.
`plot_pareto` needs the `plot` extra (`pip install -e ".[plot]"`).
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_results(path: Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text())


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
        "n_runs": len(results),
        "mean_latency_seconds": mean(latencies) if latencies else None,
        "mean_speedup": mean(speedups) if speedups else None,
        "mean_paes": mean(paes_values) if paes_values else None,
    }


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
