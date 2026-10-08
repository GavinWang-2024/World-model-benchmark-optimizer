"""Run a set of optimization combinations against the Dreamer model and print
one comparison table.

    python scripts/sweep_dreamer.py --checkpoint ../dreamerv3-torch/logdir/walker

Each configuration runs in its own process (scripts/run_dreamer.py), so
process-global switches like TF32 can't leak from one run into the next. The
baseline is the reference every speedup is measured against. Every
configuration (baseline included) runs --repeats times (default 3) and the median
latency per rollout is used, because single runs vary a lot: the eager baseline
alone has measured 223-284 ms across otherwise identical runs, so a single-run
speedup can be off by 25%. The "noise" column is the half-range of the per-run
mean latencies, as a percentage of the median — treat differences smaller than it
as unproven. Keep the GPU otherwise idle while this runs.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

from worldoptbench.reporting import drift_stats

ROOT = Path(__file__).resolve().parent.parent

CONFIGS: dict[str, list[str]] = {
    "baseline": [],
    "cuda_graphs": ["--cuda-graphs"],
    "lean_scan": ["--lean-scan"],
    "lean_scan + cuda_graphs": ["--lean-scan", "--cuda-graphs"],
    "tf32": ["--tf32"],
    "tf32 + cuda_graphs": ["--tf32", "--cuda-graphs"],
    "bf16": ["--precision", "bfloat16"],
    "bf16 + cuda_graphs": ["--precision", "bfloat16", "--cuda-graphs"],
    "fp16 + cuda_graphs": ["--precision", "float16", "--cuda-graphs"],
    "int8 weight-only": ["--quantize", "int8_weight_only"],
    "int8 + cuda_graphs": ["--quantize", "int8_weight_only", "--cuda-graphs"],
    "lean + tf32 + graphs": ["--lean-scan", "--tf32", "--cuda-graphs"],
    "no_dist_validation": ["--no-dist-validation"],
    "gumbel": ["--gumbel"],
    "gumbel + cuda_graphs": ["--gumbel", "--cuda-graphs"],
    "tensorrt": ["--tensorrt"],
    "tensorrt + cuda_graphs": ["--tensorrt", "--cuda-graphs"],
    "lean + bf16 + graphs": ["--lean-scan", "--precision", "bfloat16", "--cuda-graphs"],
}


# Built and unit-tested but never measured. Run with --experimental. Each is stacked on the best
# measured configuration (tensorrt + cuda_graphs) so its effect is read against that, not against eager.
_BEST = ["--tensorrt", "--cuda-graphs"]
EXPERIMENTAL: dict[str, list[str]] = {
    "tensorrt + cuda_graphs": _BEST,  # the reference every row below is compared against
    "cuda_graphs + share_pool": ["--cuda-graphs", "--share-graph-pool"],
    "cuda_graphs + cudnn_benchmark": ["--cuda-graphs", "--cudnn-benchmark"],
    "cuda_graphs + channels_last": ["--cuda-graphs", "--channels-last"],
    "cuda_graphs + low_rank 0.25": ["--cuda-graphs", "--low-rank", "0.25"],
    "cuda_graphs + low_rank 0.5": ["--cuda-graphs", "--low-rank", "0.5"],
    "trt fp16 + cuda_graphs": ["--tensorrt", "--tensorrt-fp16", "--cuda-graphs"],
    "trt + trt_decoder + graphs": ["--tensorrt", "--tensorrt-decoder", "--cuda-graphs"],
    "trt + sparse_decode 2 + graphs": [*_BEST, "--sparse-decode", "2"],
    "trt + sparse_decode 4 + graphs": [*_BEST, "--sparse-decode", "4"],
    "trt + sparse_decode 8 + graphs": [*_BEST, "--sparse-decode", "8"],
    "trt + latent_noise 0 + graphs": [*_BEST, "--latent-noise", "0"],
    "trt + latent_noise 0.5 + graphs": [*_BEST, "--latent-noise", "0.5"],
    "trt + trt_decoder + sparse 4": ["--tensorrt", "--tensorrt-decoder", "--cuda-graphs", "--sparse-decode", "4"],
}


def _key(r: dict) -> tuple[str, float]:
    return (r["prompt_id"], r["horizon"])


def merge_repeats(runs: list[list[dict]]) -> list[dict]:
    """One row per rollout with the *median* latency across repeats (physics and
    visual metrics come from the first repeat; they don't depend on timing)."""
    merged = []
    for i, first in enumerate(runs[0]):
        row = json.loads(json.dumps(first))
        row["speed"]["latency_seconds"] = st.median(run[i]["speed"]["latency_seconds"] for run in runs)
        merged.append(row)
    return merged


def summarize(runs: list[list[dict]], merged: list[dict], baseline: list[dict]) -> dict:
    base_lat = {_key(r): r["speed"]["latency_seconds"] for r in baseline}
    base_skill = {_key(r): r["physics"]["sim_fidelity_score"] for r in baseline}
    skills = {_key(r): r["physics"]["sim_fidelity_score"] for r in merged}
    speedups = [base_lat[_key(r)] / r["speed"]["latency_seconds"] for r in merged]
    # PAES is linear in speedup, so rebuild it with the merged speedup instead of the
    # single-run one the runner computed.
    paes = [
        (r["paes"] / r["speed"]["speedup"]) * sp if r["speed"]["speedup"] else 0.0
        for r, sp in zip(merged, speedups)
    ]
    per_run_mean = [st.mean(r["speed"]["latency_seconds"] for r in run) for run in runs]
    return {
        "speedup": st.mean(speedups),
        "latency_ms": 1000 * st.mean(r["speed"]["latency_seconds"] for r in merged),
        "noise_pct": 100 * (max(per_run_mean) - min(per_run_mean)) / 2 / st.median(per_run_mean),
        "skill": st.mean(skills.values()),
        "d_skill": st.mean(skills.values()) - st.mean(base_skill.values()),
        "max_d_skill": max(abs(skills[k] - base_skill[k]) for k in skills),
        "psnr": st.mean(r["visual"]["psnr"] for r in merged),
        "paes": st.mean(paes),
        "reserved_mb": 1024 * max(r["speed"].get("vram_reserved_gb") or 0 for r in merged),
        **drift_stats(merged),
    }


def run_once(checkpoint: Path, out: Path, baseline: Path | None, flags: list[str], standard_set: Path | None = None, extra: list[str] | None = None) -> list[dict] | None:
    cmd = [sys.executable, str(ROOT / "scripts" / "run_dreamer.py"), "--checkpoint", str(checkpoint), "--out", str(out)]
    if baseline is not None:
        cmd += ["--baseline", str(baseline)]
    if standard_set is not None:
        cmd += ["--standard-set", str(standard_set)]
    cmd += flags + (extra or [])
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)  # a failed config is reported, not fatal
    if proc.returncode != 0:
        last = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()][-3:]
        print(f"  FAILED ({proc.returncode}): {' | '.join(last)[:300]}")
        return None
    return json.loads(out.read_text(encoding="utf-8"))


def _fmt_drift(m: dict) -> str:
    if m["mean_drift_rate"] is None:
        return "n/a"
    sem = f" +- {m['drift_rate_sem']:.4f}" if m["drift_rate_sem"] is not None else ""
    flag = {True: "  detectable", False: "  not distinguishable from 0", None: ""}[m["drift_detectable"]]
    return f"{m['mean_drift_rate']:+.4f}{sem}{flag}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results" / "sweep")
    ap.add_argument("--only", nargs="*", help="run just these configuration names (baseline always runs)")
    ap.add_argument("--episodes", type=Path, default=None, metavar="DIR",
                    help="score against held-out recorded episodes (e.g. <logdir>/eval_eps) instead of random-action scenarios")
    ap.add_argument("--min-return", type=float, default=None, help="with --episodes: only episodes with at least this return")
    ap.add_argument("--experimental", action="store_true",
                    help="run the experimental (unmeasured) configurations instead of the standard ones")
    ap.add_argument("--repeats", type=int, default=3, help="runs per configuration; medians are reported")
    ap.add_argument("--standard-set", type=Path, default=None,
                    help="prompt set JSON (default: the 3-scenario set; use prompts/dreamer_dmc_set_10seeds.json to measure drift)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    extra: list[str] = []
    if args.episodes:
        extra += ["--episodes", str(args.episodes)]
        if args.min_return is not None:
            extra += ["--min-return", str(args.min_return)]

    def file_stem(name: str) -> str:
        return name.replace(" ", "").replace("+", "_")

    print(f"baseline x{args.repeats} ...", flush=True)
    base_runs = [r for i in range(args.repeats) if (r := run_once(args.checkpoint, args.out_dir / f"baseline_{i}.json", None, [], args.standard_set, extra)) is not None]
    if not base_runs:
        print("baseline failed; aborting")
        return 1
    baseline_rows = merge_repeats(base_runs)
    baseline_path = args.out_dir / "baseline.json"
    baseline_path.write_text(json.dumps(baseline_rows, indent=2), encoding="utf-8")

    summaries = {"baseline": summarize(base_runs, baseline_rows, baseline_rows)}
    table = EXPERIMENTAL if args.experimental else CONFIGS
    names = [n for n in table if n != "baseline" and (not args.only or n in args.only)]
    for name in names:
        print(f"{name} x{args.repeats} ...", flush=True)
        runs = []
        for i in range(args.repeats):
            rows = run_once(args.checkpoint, args.out_dir / f"{file_stem(name)}_{i}.json", baseline_path, table[name], args.standard_set, extra)
            if rows is None:
                break
            runs.append(rows)
        if len(runs) == args.repeats:
            merged = merge_repeats(runs)
            (args.out_dir / f"{file_stem(name)}.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
            summaries[name] = summarize(runs, merged, baseline_rows)

    print()
    print(f"{'configuration':26s} {'speedup':>8s} {'lat(ms)':>8s} {'noise':>6s} {'skill':>6s} {'d_skill':>8s} {'max|d|':>7s} {'psnr':>6s} {'PAES':>6s} {'held MB':>8s} {'drift /s (mean +- sem)':>24s}")
    for name, m in summaries.items():
        print(f"{name:26s} {m['speedup']:8.2f} {m['latency_ms']:8.1f} {m['noise_pct']:5.0f}% {m['skill']:6.3f} {m['d_skill']:+8.4f} {m['max_d_skill']:7.3f} {m['psnr']:6.2f} {m['paes']:6.2f} {m['reserved_mb']:8.0f} {_fmt_drift(m):>24s}")
    (args.out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
