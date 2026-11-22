"""V-JEPA 2 (Meta) as a latent-forecasting world model behind WorldModelInterface.

V-JEPA 2 predicts *representations*, not pixels: there is no decoder, so the frame-based visual and physics
metrics cannot score it, and `generate` returns latents, not frames. What it can do, and what this wrapper
exposes, is the model's actual job: encode a context clip, then predict the encoder's latents for the
tubelet-steps that follow. Quality for this model is therefore latent-forecast fidelity (`metrics/latent.py`),
a different axis from PSNR/SSIM and not comparable with the Dreamer or diffusion numbers.

No future leakage. `transformers`' `VJEPA2Model.forward` runs the encoder over the *whole* clip and applies the
masks afterwards, so its "context" tokens have already attended to the future frames. A forecast made that way is
not a forecast. Here the encoder sees only the context frames, and the predictor is called directly on those
tokens with target positions that lie beyond them. The true future latents (for scoring) come from a separate,
full-clip pass, `encode_clip`, which is deliberately not part of `generate`.

Token layout, checked against the installed `transformers` source: the Conv3d patch embedding flattens to
(time-step, row, column) with time slowest, so tubelet-step `s` occupies token indices
[s * tokens_per_step, (s + 1) * tokens_per_step). The position encoding is rotary, so the context may be any
number of tubelet-steps (the config's `frames_per_clip` is documented as not affecting inference).
"""

from __future__ import annotations

from typing import Any

from worldoptbench.models.base import ModelInfo, Rollout, WorldModelInterface

REPO_ID = "facebook/vjepa2-vitl-fpc64-256"
# From the checkpoint's video_preprocessor_config.json: ImageNet statistics, shortest edge resized to 292
# (= 256 * 1.14), then a center crop to the model's input size.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_FACTOR = 292 / 256
# V-JEPA 2 was trained on 4 fps clips (fpc64 = 16 s); only used to turn a horizon in seconds into steps.
DEFAULT_FPS = 4.0


def preprocess_video(frames: Any, crop_size: int):
    """(T, H, W, 3) uint8/float frames (numpy or torch) -> (1, T, 3, crop, crop) normalized float32 tensor.

    Resize the shortest edge to crop_size * 292/256 (bilinear), center crop, scale to [0, 1] when the input
    is 0-255, then apply the ImageNet normalization. Mirrors the checkpoint's own video processor without
    depending on its optional video-decoding packages.
    """
    import torch
    import torch.nn.functional as F

    x = torch.as_tensor(frames)
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"expected frames of shape (T, H, W, 3), got {tuple(x.shape)}")
    x = x.permute(0, 3, 1, 2).float()  # (T, 3, H, W)
    if x.max() > 1.5:  # 0-255 input
        x = x / 255.0
    _, _, h, w = x.shape
    short = round(crop_size * RESIZE_FACTOR)
    scale = short / min(h, w)
    new_h, new_w = max(crop_size, round(h * scale)), max(crop_size, round(w * scale))
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
    top, left = (new_h - crop_size) // 2, (new_w - crop_size) // 2
    x = x[:, :, top : top + crop_size, left : left + crop_size]
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype).view(1, 3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)


def horizon_to_steps(horizon: float, fps: float, tubelet_size: int) -> int:
    """Future tubelet-steps for a horizon in seconds (at least one)."""
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    return max(1, round(horizon * fps / tubelet_size))


class VJEPA2Latent(WorldModelInterface):
    """V-JEPA 2 encoder + predictor. `generate` forecasts future latents from a context clip.

    Optimization hooks (all optional capabilities from `models.base`): `torch_module()` exposes the whole
    `VJEPA2Model` so architecture-agnostic modules (quantization, ...) can modify it in place, and
    `autocast_dtype` lets the precision module run the forward under `torch.autocast`.
    """

    def __init__(
        self,
        repo_id: str = REPO_ID,
        device: str | None = None,
        torch_dtype: Any = None,
        fps: float = DEFAULT_FPS,
        model: Any = None,
    ):
        """
        Args:
            repo_id: Hugging Face checkpoint (public; no token or license click-through needed).
            device: "cuda" / "cpu"; default is CUDA when available.
            torch_dtype: weight dtype; default float32, so the unoptimized baseline is full precision.
            fps: frame rate the horizon in seconds is converted with (V-JEPA 2 trained at 4 fps).
            model: an already-built `VJEPA2Model` (used by the tests); skips the download.
        """
        self.repo_id = repo_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.fps = fps
        self.autocast_dtype: Any = None  # set by the precision module; None = run in the weights' dtype
        self._model = model

    # -- loading ----------------------------------------------------------------------------------------

    def _resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def torch_module(self) -> Any:
        """The loaded `VJEPA2Model` (downloaded on first use) — the `TorchBacked` capability."""
        if self._model is None:
            import torch
            from transformers import VJEPA2Model

            model = VJEPA2Model.from_pretrained(self.repo_id, torch_dtype=self.torch_dtype or torch.float32)
            self._model = model.to(self._resolve_device()).eval()
        return self._model

    @property
    def config(self) -> Any:
        return self.torch_module().config

    @property
    def tokens_per_step(self) -> int:
        cfg = self.config
        return (cfg.crop_size // cfg.patch_size) ** 2

    def get_info(self) -> ModelInfo:
        model = self.torch_module()
        return ModelInfo(
            name="V-JEPA 2 (ViT-L, latent forecast)",
            architecture="jepa",
            param_count=sum(p.numel() for p in model.parameters()),
            supported_optimizations=["precision", "quantization", "tf32", "cudnn_benchmark", "low_rank"],
            supports_image_conditioning=False,
            supports_video_conditioning=True,
        )

    # -- forward helpers --------------------------------------------------------------------------------

    def _autocast(self):
        import contextlib

        import torch

        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=torch.device(self._resolve_device()).type, dtype=self.autocast_dtype)

    def prepare(self, frames: Any):
        """Raw (T, H, W, 3) frames -> the model's input tensor (1, T, 3, crop, crop) on the model's device.

        `generate` and `encode_clip` also accept this tensor directly, which lets a benchmark time the model
        without the (CPU, constant, ~50 ms for 32 frames) resize-and-normalize.
        """
        model = self.torch_module()
        param = next(model.parameters())
        return preprocess_video(frames, model.config.crop_size).to(device=param.device, dtype=param.dtype)

    def _pixels(self, frames: Any):
        import torch

        if isinstance(frames, torch.Tensor) and frames.ndim == 5:  # already prepared
            return frames
        return self.prepare(frames)

    def encode_clip(self, frames: Any):
        """Encoder latents for a whole clip: (n_steps * tokens_per_step, hidden), float32 on the CPU.

        This is the scoring target (what the encoder says the future actually looks like). It is not part
        of `generate`, so its cost never lands in a forecast's latency.
        """
        import torch

        model = self.torch_module()
        pixels = self._pixels(frames)
        self._check_frame_count(pixels.shape[1])
        with torch.no_grad(), self._autocast():
            out = model.encoder(pixel_values_videos=pixels).last_hidden_state
        return out[0].float().cpu()

    def _check_frame_count(self, n_frames: int) -> int:
        tubelet = self.config.tubelet_size
        if n_frames < tubelet or n_frames % tubelet:
            raise ValueError(f"frame count must be a positive multiple of the tubelet size {tubelet}, got {n_frames}")
        return n_frames // tubelet

    # -- the interface ----------------------------------------------------------------------------------

    def generate(
        self,
        prompt: str | None = None,
        init_frame: Any | None = None,
        init_video: Any | None = None,
        actions: list[Any] | None = None,
        horizon: float = 4.0,
        **kwargs: Any,
    ) -> Rollout:
        """Forecast the latents of the next `horizon` seconds from a context clip (Video2World).

        Args:
            init_video: the context, (T, H, W, 3) frames; T must be a multiple of the tubelet size (2). A
                tensor already returned by `prepare` (5-D) is used as is.
            horizon: seconds to forecast (converted to tubelet-steps at `fps`).
            future_steps: optional kwarg overriding the horizon with an exact number of tubelet-steps.
            prompt, init_frame, actions: unsupported. V-JEPA 2 has no text or image conditioning, and this is
                the base predictor, not the action-conditioned V-JEPA 2-AC variant, so they raise rather than
                being silently ignored.

        Returns:
            A `Rollout` whose `frames` are the predicted latents, one (tokens_per_step, hidden) float32
            array per future tubelet-step (NOT images). `metadata["kind"] == "latent"`, and it holds the
            last context step's latents (`context_last`, the persistence baseline) and the step counts.
        """
        import torch

        if init_video is None:
            raise ValueError("V-JEPA 2 needs init_video: a context clip to forecast from")
        if prompt is not None or init_frame is not None:
            raise ValueError("V-JEPA 2 has no text or image conditioning; pass init_video only")
        if actions is not None:
            raise ValueError("actions are not supported (this is the base predictor, not V-JEPA 2-AC)")

        model = self.torch_module()
        pixels = self._pixels(init_video)
        n_ctx = self._check_frame_count(pixels.shape[1])
        n_future = int(kwargs.pop("future_steps", 0)) or horizon_to_steps(horizon, self.fps, model.config.tubelet_size)
        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
        per_step = self.tokens_per_step
        device = pixels.device

        # Positions only: the context tokens are 0..n_ctx*per_step, the targets follow immediately after.
        context_mask = [torch.arange(0, n_ctx * per_step, device=device).unsqueeze(0)]
        target_mask = [torch.arange(n_ctx * per_step, (n_ctx + n_future) * per_step, device=device).unsqueeze(0)]
        with torch.no_grad(), self._autocast():
            context = model.encoder(pixel_values_videos=pixels).last_hidden_state  # the encoder sees ONLY the context
            predicted = model.predictor(
                encoder_hidden_states=context, context_mask=context_mask, target_mask=target_mask
            ).last_hidden_state
        predicted = predicted[0].float().cpu()
        last_context = context[0, -per_step:].float().cpu()
        return Rollout(
            frames=[predicted[s * per_step : (s + 1) * per_step].numpy() for s in range(n_future)],
            fps=self.fps / model.config.tubelet_size,
            metadata={
                "kind": "latent",
                "context_steps": n_ctx,
                "future_steps": n_future,
                "tokens_per_step": per_step,
                "context_last": last_context.numpy(),
            },
        )
