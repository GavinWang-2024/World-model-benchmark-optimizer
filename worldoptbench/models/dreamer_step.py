"""One RSSM imagination step written for export, plus a pure-PyTorch imagination
backend built from it.

Imports torch at module level, so import this lazily (the rest of
`worldoptbench.models.dreamer` deliberately stays importable without torch).

Why a rewrite of the repo's `RSSM.img_step`: the original samples with
`torch.multinomial`, which no inference engine (TensorRT, ONNX) can express, and
spends ~70 tiny kernels per step on it (distribution validation, mixing,
sampling, straight-through bookkeeping). This version computes the same math
with exportable ops and moves the randomness out of the step:

    logits   -> softmax -> unimix  -> log
    sample   =  argmax(log_probs + Gumbel noise)   # Gumbel-max trick

Gumbel-max draws from exactly the same categorical distribution as
`torch.multinomial`, but uses a different random stream, so rollouts are
statistically equivalent to the repo's, not sample-for-sample identical. The
deterministic part (new hidden state) matches the repo's `img_step` exactly
(max abs difference 0.0 on the trained walker; verified in the prototype).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def gumbel_noise(shape: tuple[int, ...], device: Any) -> torch.Tensor:
    """Standard Gumbel(0, 1) noise. `u` is clamped away from 0 and 1 so both
    logs stay finite — an unclamped version produced NaNs and silently wrong
    samples in an early prototype.
    """
    u = torch.rand(shape, device=device).clamp(1e-9, 1.0 - 1e-9)
    return -torch.log(-torch.log(u))


class RSSMStepCore(nn.Module):
    """One imagination step: (stoch, action, deter, gumbel) -> (stoch', deter').

    Shares weights with the RSSM it wraps (it holds a reference, not a copy), so
    it always reflects the loaded checkpoint and any in-place quantization that
    happened before it ran. Discrete latents with `rec_depth == 1` only — the
    Dreamer default; anything else raises rather than silently differing.
    """

    def __init__(self, dyn: Any):
        super().__init__()
        if not dyn._discrete:
            raise NotImplementedError("RSSMStepCore supports discrete latents only")
        if dyn._rec_depth != 1:
            raise NotImplementedError("RSSMStepCore supports rec_depth == 1 only")
        self.dyn = dyn
        self.stoch_size = int(dyn._stoch)
        self.classes_n = int(dyn._discrete)
        self.unimix = float(dyn._unimix_ratio)
        device = next(dyn.parameters()).device
        self.register_buffer(
            "classes", torch.arange(self.classes_n, device=device).view(1, 1, self.classes_n), persistent=False
        )

    def forward(self, stoch: torch.Tensor, action: torch.Tensor, deter: torch.Tensor, gumbel: torch.Tensor):
        d = self.dyn
        batch = stoch.shape[0]
        x = d._img_in_layers(torch.cat([stoch.reshape(batch, -1), action], -1))
        out, state = d._cell(x, [deter])
        new_deter = state[0]
        logit = d._imgs_stat_layer(d._img_out_layers(out)).reshape(batch, self.stoch_size, self.classes_n)
        probs = torch.softmax(logit, -1) * (1.0 - self.unimix) + self.unimix / self.classes_n
        index = torch.argmax(torch.log(probs) + gumbel, -1, keepdim=True)
        return (self.classes == index).to(logit.dtype), new_deter


class TorchGumbelImagination:
    """Imagination backend (see `models.base.HasImagineBackend`): the step above
    in a plain PyTorch loop. Serves as the exact reference for the TensorRT
    backend (same noise, same math) and is itself a little faster than the
    repo's loop because it skips distribution objects and validation.
    """

    def __init__(self) -> None:
        self._cores: dict[int, RSSMStepCore] = {}
        # Multiplies the Gumbel noise: 1.0 samples the true distribution, 0.0 takes
        # the most likely class every step (a deterministic "mode" rollout), values
        # in between trade diversity for fidelity. See optimizations/latent_noise.py.
        self.noise_scale = 1.0

    def _core(self, wm: Any) -> RSSMStepCore:
        core = self._cores.get(id(wm))
        if core is None:
            core = self._cores[id(wm)] = RSSMStepCore(wm.dynamics).eval()
        return core

    def draw_noise(self, wm: Any, stoch: Any, steps: int) -> Any:
        """The Gumbel noise for `steps` imagination steps, shape (steps, B, stoch, classes), already scaled.
        Drawing it separately from the loop lets a chunked rollout draw everything up front, so its random
        stream does not depend on the chunk size."""
        core = self._core(wm)
        return gumbel_noise((steps, stoch.shape[0], core.stoch_size, core.classes_n), stoch.device) * self.noise_scale

    def __call__(self, wm: Any, stoch: Any, deter: Any, future_actions: Any, noise: Any = None) -> dict[str, Any]:
        core = self._core(wm)
        steps = future_actions.shape[1]
        if noise is None:
            noise = self.draw_noise(wm, stoch, steps)
        stochs, deters = [], []
        for t in range(steps):
            stoch, deter = core(stoch, future_actions[:, t], deter, noise[t])
            stochs.append(stoch)
            deters.append(deter)
        return {"stoch": torch.stack(stochs, 1), "deter": torch.stack(deters, 1)}


class DecoderCore(nn.Module):
    """The Dreamer image decoder as a bare tensor function, for export:
    latent features (B, T, F) -> float images (B, T, H, W, C) in [0, 1].

    This is the convolutional head's output directly — the same tensor the repo's
    `decoder(feat)["image"].mode()` returns for its default MSE image distribution —
    without the distribution wrapper, which an exporter can't see through.
    """

    def __init__(self, decoder: Any):
        super().__init__()
        if not hasattr(decoder, "_cnn"):
            raise NotImplementedError("DecoderCore needs a decoder with a convolutional head (`_cnn`)")
        self.cnn = decoder._cnn

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.cnn(features)
