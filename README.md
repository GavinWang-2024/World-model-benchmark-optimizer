# WorldOptBench

Architecture-agnostic benchmarking and optimization framework for world model inference — joint speed, visual quality, and physics consistency (see `PAES`).

- **Full spec / pitch doc:** [`world_model_project_outline.md`](world_model_project_outline.md)
- **Execution checklist (what to build in what order):** [`build_plan.md`](build_plan.md)

## Setup

```bash
pip install -e ".[dev]"
pytest
```

GPU/model work needs the `ml` extra and a hardware-appropriate Torch/CUDA build — install Torch yourself first per your rented/HPC box, then:

```bash
pip install -e ".[ml]"
```

See `build_plan.md` Phase 0 for HF access + compute setup.
