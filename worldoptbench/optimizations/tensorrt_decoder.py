"""TensorRT for the image decoder.

After CUDA graphs and a TensorRT imagination step, decoding is the largest single
piece left (~5 of ~13 ms at a 20 s horizon: 13 us per frame). This exports the
decoder's convolutional head to ONNX, builds a TensorRT engine for the feature
shape in play, and runs it in place of the PyTorch decoder call. Unlike the step
engine it involves no sampling, so it should match the PyTorch decoder up to
floating-point differences rather than only statistically.

Engines are static-shape: one is built (and cached on disk, keyed by the exported
graph and TensorRT version) per distinct (batch, frames, features) shape, so each
horizon gets its own — the first call at each shape pays ~seconds, hence the
runner's per-horizon warm-up. Like the step engine, TensorRT's own workspace is
outside PyTorch's allocator and isn't in the reported GPU memory.

Needs the `tensorrt-cu12` and `onnx` packages (never `torch-tensorrt` next to an
older torch). Measured on the trained walker: the ONNX export of the decoder's custom layers worked first time, the output is IDENTICAL to
the PyTorch decoder (zero difference in skill and PSNR across every rollout), and it is ~8-10% faster end to end (12.0 vs 13.1 ms
at the sweep's mix of horizons; 23.8 vs 26.2 ms at 20 s). Less than hoped, because the decoder was already a small share.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

from worldoptbench.models.base import Architecture, WorldModelInterface
from worldoptbench.optimizations.base import OptimizationModule, register_module


class TensorRTDecoder:
    """Decode backend (`DreamerWorldModel.decode_backend`): `(wm, features) -> images`."""

    def __init__(self, cache_dir: str | Path | None = None):
        self._cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "worldoptbench" / "trt"
        self._engines: dict[Any, tuple[Any, tuple[int, ...]]] = {}

    def _build(self, wm: Any, features: Any) -> tuple[Any, tuple[int, ...]]:
        import tensorrt as trt
        import torch

        from worldoptbench.models.dreamer_step import DecoderCore

        core = DecoderCore(wm.heads["decoder"]).eval()
        example = features.detach().contiguous()
        buffer = io.BytesIO()
        with torch.no_grad(), torch.autocast(device_type=example.device.type, enabled=False):
            torch.onnx.export(
                core, (example,), buffer,
                input_names=["features"], output_names=["image"],
                opset_version=17, dynamo=False,
            )
        onnx_bytes = buffer.getvalue()

        logger = trt.Logger(trt.Logger.ERROR)
        key = hashlib.sha1(
            onnx_bytes + trt.__version__.encode() + torch.cuda.get_device_name(example.device).encode()
        ).hexdigest()
        cached = self._cache_dir / f"decoder_{key}.engine"
        runtime = trt.Runtime(logger)
        if cached.exists():
            engine = runtime.deserialize_cuda_engine(cached.read_bytes())
        else:
            builder = trt.Builder(logger)
            network = builder.create_network()
            parser = trt.OnnxParser(network, logger)
            if not parser.parse(onnx_bytes):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError(f"TensorRT could not parse the exported decoder: {errors}")
            serialized = builder.build_serialized_network(network, builder.create_builder_config())
            if serialized is None:
                raise RuntimeError("TensorRT failed to build an engine for the decoder")
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(bytes(serialized))
            engine = runtime.deserialize_cuda_engine(bytes(serialized))
        return engine.create_execution_context(), tuple(engine.get_tensor_shape("image"))

    def __call__(self, wm: Any, features: Any) -> Any:
        import torch

        key = (id(wm), tuple(features.shape), features.dtype)
        entry = self._engines.get(key)
        if entry is None:
            entry = self._engines[key] = self._build(wm, features)
        context, out_shape = entry

        features = features.contiguous()
        out = torch.empty(out_shape, device=features.device, dtype=torch.float32)
        context.set_tensor_address("features", features.data_ptr())
        context.set_tensor_address("image", out.data_ptr())
        context.execute_async_v3(torch.cuda.current_stream().cuda_stream)  # the capture stream under CUDA graphs
        return out


@register_module
class TensorRTDecoderModule(OptimizationModule):
    name = "tensorrt_decoder"
    requires = ("decode_backend",)
    needs_cuda = True
    needs_packages = ("tensorrt", "onnx")
    maturity = "measured"
    summary = "Image decoder as a TensorRT engine. Measured: identical output (0 difference in skill/PSNR), ~8-10% faster."
    supported_architectures: tuple[Architecture, ...] = ("autoregressive",)

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = cache_dir

    def apply(self, model: WorldModelInterface) -> WorldModelInterface:
        if not hasattr(model, "decode_backend"):
            raise TypeError(
                f"{type(model).__name__} has no `decode_backend` hook, so there is no decoder call to replace"
            )
        try:
            import tensorrt  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "TensorRTDecoderModule needs tensorrt: `pip install tensorrt-cu12 onnx` "
                "(don't install torch-tensorrt next to an older torch — it replaces it)"
            ) from e
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("tensorrt_decoder needs a CUDA device; none is available")
        model.decode_backend = TensorRTDecoder(self.cache_dir)
        return model
