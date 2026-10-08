"""TensorRT for the RSSM imagination step.

The imagination loop is a chain of ~60 tiny kernels per generated frame. CUDA
graphs (cuda_graphs.py) removed the *launch* overhead; what's left is the chain
itself, and TensorRT fuses that chain into far fewer kernels. Prototype result
on the trained Dreamer walker (RTX 5070 Ti, TensorRT 11.3, 400 steps, sampling
included, both inside a CUDA graph): 36 us/step vs 62 us/step for PyTorch, i.e.
1.7x on the loop, with the sampled states agreeing 100% over all 400 chained
steps against the PyTorch reference given the same noise.

How it works: the one-step math in `models/dreamer_step.RSSMStepCore` is exported
to ONNX, built into a TensorRT engine (cached on disk, ~5 s to build), and run
once per step on PyTorch's current CUDA stream — so it composes with
`cuda_graphs`, which can capture the TensorRT launches. Sampling is Gumbel-max
with noise drawn up front in PyTorch (TensorRT has no RNG), so outputs are
statistically equivalent to the repo's `torch.multinomial` sampling but not
sample-for-sample identical; compare against `gumbel_sampling` (same noise, pure
PyTorch) for an exact check.

Requirements / limits:
  - `pip install tensorrt-cu12 onnx` (with numpy pinned if your env needs it —
    their default resolution wants numpy 2.x). Do NOT install `torch-tensorrt`
    alongside an older torch: it pins a matching torch and replaces yours.
  - batch size 1 and the engine is static-shape per (batch); discrete latents,
    rec_depth 1 (anything else raises).
  - The engine is fp32 (TensorRT 11 is strongly typed: precision follows the
    exported tensor types), so `precision`/`tf32` modules don't affect it.
  - TensorRT allocates its own workspace outside PyTorch's allocator, so the
    reported GPU memory (`vram_reserved_gb`) understates the real footprint.
"""

from __future__ import annotations

import copy
import hashlib
import io
from pathlib import Path
from typing import Any

from worldoptbench.models.base import Architecture, WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, register_module

_HAS_BACKEND = "imagine_backend"


class TensorRTImagination:
    """Imagination backend (`models.base.HasImagineBackend`) running the RSSM
    step as a TensorRT engine. Engines are built lazily on first use — during
    the runner's warm-up, i.e. outside any CUDA-graph capture — and cached per
    model (and on disk, keyed by the exported graph + TensorRT version).
    """

    def __init__(self, cache_dir: str | Path | None = None, precision: str = "fp32"):
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"precision must be 'fp32' or 'fp16', got {precision!r}")
        self.precision = precision
        self.noise_scale = 1.0  # see models/dreamer_step.TorchGumbelImagination
        self._cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "worldoptbench" / "trt"
        self._contexts: dict[int, Any] = {}

    def _build_context(self, wm: Any, batch: int) -> Any:
        import tensorrt as trt
        import torch

        from worldoptbench.models.dreamer_step import RSSMStepCore, gumbel_noise

        core = RSSMStepCore(wm.dynamics).eval()
        dtype = torch.float32
        if self.precision == "fp16":
            # A deep copy: `.half()` converts parameters in place, and the core holds the
            # model's own RSSM, which the (fp32) context phase still needs.
            core = copy.deepcopy(core).half()
            dtype = torch.float16
        device = next(wm.parameters()).device
        s, c = core.stoch_size, core.classes_n
        example = (
            torch.zeros(batch, s, c, device=device, dtype=dtype),
            torch.zeros(batch, int(wm.dynamics._num_actions), device=device, dtype=dtype),
            torch.zeros(batch, int(wm.dynamics._deter), device=device, dtype=dtype),
            gumbel_noise((batch, s, c), device).to(dtype),
        )
        buffer = io.BytesIO()
        # autocast disabled: an export traced inside an active autocast region would bake in
        # whatever mixed precision happened to be on, instead of the precision requested here.
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            torch.onnx.export(
                core, example, buffer,
                input_names=["stoch", "action", "deter", "gumbel"],
                output_names=["stoch_out", "deter_out"],
                opset_version=17, dynamo=False,
            )
        onnx_bytes = buffer.getvalue()

        logger = trt.Logger(trt.Logger.ERROR)
        key = hashlib.sha1(onnx_bytes + trt.__version__.encode() + torch.cuda.get_device_name(device).encode()).hexdigest()
        cached = self._cache_dir / f"{key}.engine"
        runtime = trt.Runtime(logger)
        if cached.exists():
            engine = runtime.deserialize_cuda_engine(cached.read_bytes())
        else:
            builder = trt.Builder(logger)
            network = builder.create_network()
            parser = trt.OnnxParser(network, logger)
            if not parser.parse(onnx_bytes):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError(f"TensorRT could not parse the exported RSSM step: {errors}")
            serialized = builder.build_serialized_network(network, builder.create_builder_config())
            if serialized is None:
                raise RuntimeError("TensorRT failed to build an engine for the RSSM step")
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(bytes(serialized))
            engine = runtime.deserialize_cuda_engine(bytes(serialized))
        return engine.create_execution_context()

    def draw_noise(self, wm: Any, stoch: Any, steps: int) -> Any:
        """The Gumbel noise for `steps` steps, shape (steps, B, stoch, classes), scaled and in the engine's
        working dtype. Drawn apart from the loop so a chunked rollout can draw it all up front."""
        import torch

        from worldoptbench.models.dreamer_step import gumbel_noise

        s, c = stoch.shape[1:]
        work = torch.float16 if self.precision == "fp16" else stoch.dtype
        return (gumbel_noise((steps, stoch.shape[0], s, c), stoch.device) * self.noise_scale).to(work)

    def __call__(self, wm: Any, stoch: Any, deter: Any, future_actions: Any, noise: Any = None) -> dict[str, Any]:
        import torch

        batch, steps = future_actions.shape[:2]
        if batch != 1:
            raise NotImplementedError("the TensorRT backend is built for batch size 1")
        context = self._contexts.get(id(wm))
        if context is None:
            context = self._contexts[id(wm)] = self._build_context(wm, batch)

        s, c = stoch.shape[1:]
        work = torch.float16 if self.precision == "fp16" else stoch.dtype
        if noise is None:
            noise = self.draw_noise(wm, stoch, steps)
        actions = future_actions.transpose(0, 1).contiguous().to(work)  # (N, B, A): one contiguous slice per step
        stoch_buf = torch.empty(steps + 1, batch, s, c, device=stoch.device, dtype=work)
        deter_buf = torch.empty(steps + 1, batch, deter.shape[1], device=deter.device, dtype=work)
        stoch_buf[0], deter_buf[0] = stoch, deter

        stream = torch.cuda.current_stream().cuda_stream  # the capture stream, under CUDA graphs
        for t in range(steps):
            context.set_tensor_address("stoch", stoch_buf[t].data_ptr())
            context.set_tensor_address("action", actions[t].data_ptr())
            context.set_tensor_address("deter", deter_buf[t].data_ptr())
            context.set_tensor_address("gumbel", noise[t].data_ptr())
            context.set_tensor_address("stoch_out", stoch_buf[t + 1].data_ptr())
            context.set_tensor_address("deter_out", deter_buf[t + 1].data_ptr())
            context.execute_async_v3(stream)
        # Back to the model's dtype for the decoder, whatever the engine ran in.
        return {
            "stoch": stoch_buf[1:].transpose(0, 1).to(stoch.dtype),
            "deter": deter_buf[1:].transpose(0, 1).to(deter.dtype),
        }


@register_module
class TensorRTModule(OptimizationModule):
    name = "tensorrt"
    requires = ("imagine_backend",)
    needs_cuda = True
    needs_packages = ("tensorrt", "onnx")
    maturity = "measured"
    summary = "RSSM step as a TensorRT engine. Measured: fp32 13.1 ms with graphs; fp16 (precision=\"fp16\") 9.9 ms with matching physics, the fastest configuration so far."
    # Needs a recurrent latent model whose step `RSSMStepCore` can export.
    supported_architectures: tuple[Architecture, ...] = ("autoregressive",)

    def __init__(self, cache_dir: str | Path | None = None, precision: str = "fp32"):
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"precision must be 'fp32' or 'fp16', got {precision!r}")
        self.cache_dir = cache_dir
        self.precision = precision

    @property
    def label(self) -> str:
        return self.name if self.precision == "fp32" else f"{self.name}_{self.precision}"

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        if not hasattr(model, _HAS_BACKEND):
            raise TypeError(
                f"{type(model).__name__} has no `imagine_backend` hook, so there is no "
                "imagination loop to hand to TensorRT — see models.base.HasImagineBackend"
            )
        try:
            import tensorrt  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "TensorRTModule needs tensorrt: `pip install tensorrt-cu12 onnx` "
                "(don't install torch-tensorrt next to an older torch — it replaces it)"
            ) from e
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("tensorrt needs a CUDA device; none is available")
        backend = TensorRTImagination(self.cache_dir, self.precision)
        # Keep a noise scale a previous module (latent_noise) already set.
        backend.noise_scale = getattr(getattr(model, "imagine_backend", None), "noise_scale", 1.0)
        model.imagine_backend = backend
        return model
