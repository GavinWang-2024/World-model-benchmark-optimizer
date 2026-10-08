"""Learned perceptual scores for generated video: DINOv2 for "does it still look like the baseline's
content" and CLIP for "does it still match the prompt".

Why: PSNR/SSIM saturate at a noise floor on diffusion models (a bf16-level perturbation already gives
~21 dB), and the pixel statistics in `metrics/blind.py` cannot tell a different-but-fine video from a broken
one above ~0.15 (see DESIGN_DIFFUSION.md, contact-sheet check). Feature embeddings measure content:

* `dino_similarity`: mean cosine similarity between the DINOv2 embedding of each sampled frame and that of the
  reference (baseline) video's frame at the same position. High = same content in the same layout; a
  different-but-coherent scene falls somewhere in the middle; the numerical-perturbation control gives the
  scale of "same seed, equivalent computation".
* `dino_consistency` (and `dino_consistency_ref`): mean cosine similarity between consecutive sampled frames
  of the video itself; flicker, jitter and collapse into noise lower it. No reference needed.
* `clip_score` (and `clip_score_ref`, `clip_delta`): mean cosine similarity between each sampled frame and
  the prompt text under CLIP. Prompt adherence; the baseline's own score is the yardstick, since absolute CLIP
  scores are small (about 0.2-0.35) and prompt-dependent.

Models: `facebook/dinov2-small` and `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` (a safetensors CLIP; the
`openai` repo ships only a pickle). Loaded lazily on first use; ~700 MB on disk, well under 1 GB on the GPU.
Frames are sampled (default 8, evenly spaced) rather than all embedded.

Limits: general-purpose image encoders; they do not know physics, and CLIP is weak on counts, fine detail and
motion. These are better screens than pixel statistics, not ground truth about physical plausibility.

Embedding is split into `dino_embed`, `clip_image_embed` and `clip_text_embed` so tests (and other backends)
can replace them without the models.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

DINO_REPO = "facebook/dinov2-small"
CLIP_REPO = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
_IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
_CLIP = ((0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))


def sample_indices(count: int, wanted: int) -> list[int]:
    """`wanted` evenly spaced frame indices out of `count` (all of them if there are fewer)."""
    if count < 1:
        raise ValueError("a video needs at least one frame")
    return sorted({round(i) for i in np.linspace(0, count - 1, min(wanted, count))})


def cosine_rows(a: Any, b: Any) -> Any:
    """Row-wise cosine similarity of two (n, d) arrays."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)


def consecutive_consistency(embeddings: Any) -> float:
    """Mean cosine similarity between consecutive rows (1.0 for a single frame: nothing can change)."""
    embeddings = np.asarray(embeddings)
    if len(embeddings) < 2:
        return 1.0
    return float(cosine_rows(embeddings[:-1], embeddings[1:]).mean())


class PerceptualScorer:
    def __init__(self, device: str | None = None, frames_sampled: int = 8, keep_descriptor: bool = False):
        if frames_sampled < 2:
            raise ValueError("frames_sampled must be >= 2 (consistency compares consecutive frames)")
        self.frames_sampled = frames_sampled
        # keep_descriptor: also return each video's descriptor (and the reference's) so a Frechet distance between
        # sets of videos can be computed later (metrics/fvd.py); off by default because it makes results larger.
        self.keep_descriptor = keep_descriptor
        self._device = device
        self._dino: Any = None
        self._clip: Any = None
        self._tokenizer: Any = None

    # -- models --

    def _device_name(self) -> str:
        import torch

        return self._device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _load_dino(self) -> Any:
        if self._dino is None:
            from transformers import AutoModel

            self._dino = AutoModel.from_pretrained(DINO_REPO).to(self._device_name()).eval()
        return self._dino

    def _load_clip(self) -> tuple[Any, Any]:
        if self._clip is None:
            from transformers import AutoTokenizer, CLIPModel

            self._clip = CLIPModel.from_pretrained(CLIP_REPO).to(self._device_name()).eval()
            self._tokenizer = AutoTokenizer.from_pretrained(CLIP_REPO)
        return self._clip, self._tokenizer

    def _pixels(self, frames: Sequence[Any], size: tuple[int, int], stats: tuple) -> Any:
        import torch
        from torch.nn import functional

        x = torch.from_numpy(np.stack([np.ascontiguousarray(f, dtype=np.uint8) for f in frames]))
        x = x.permute(0, 3, 1, 2).float().div(255.0).to(self._device_name())
        x = functional.interpolate(x, size=size, mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
        mean = torch.tensor(stats[0], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(stats[1], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    # -- embeddings (replaceable) --

    def dino_embed(self, frames: Sequence[Any]) -> np.ndarray:
        """(n, d) DINOv2 [CLS] embeddings, frames resized (not cropped) to 224 x a multiple of 14."""
        import torch

        height, width = np.asarray(frames[0]).shape[:2]
        size = (224, max(14, round(224 * width / height / 14) * 14))
        with torch.no_grad():
            out = self._load_dino()(pixel_values=self._pixels(frames, size, _IMAGENET))
        return out.pooler_output.float().cpu().numpy()

    def clip_image_embed(self, frames: Sequence[Any]) -> np.ndarray:
        """(n, d) CLIP image embeddings; frames are squashed to 224 x 224 so no side of the frame is lost."""
        import torch

        model, _ = self._load_clip()
        with torch.no_grad():
            features = model.get_image_features(pixel_values=self._pixels(frames, (224, 224), _CLIP))
        features = getattr(features, "pooler_output", features)
        return features.float().cpu().numpy()

    def clip_text_embed(self, text: str) -> np.ndarray:
        """(1, d) CLIP text embedding."""
        import torch

        model, tokenizer = self._load_clip()
        tokens = tokenizer(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        with torch.no_grad():
            features = model.get_text_features(**{k: v.to(self._device_name()) for k, v in tokens.items()})
        features = getattr(features, "pooler_output", features)
        return features.float().cpu().numpy()

    # -- scores --

    @staticmethod
    def descriptor(embeddings: Any) -> list[float]:
        """A video-level vector from its sampled frame embeddings: the mean and the standard deviation over
        frames of the unit-normalized embeddings (what the video shows, and how much that varies in time)."""
        e = np.asarray(embeddings, dtype=np.float64)
        e = e / (np.linalg.norm(e, axis=-1, keepdims=True) + 1e-12)
        return np.concatenate([e.mean(axis=0), e.std(axis=0)]).tolist()

    def score(self, frames: Sequence[Any], reference_frames: Sequence[Any] | None = None, prompt: str | None = None) -> dict[str, Any]:
        """Scores one video. Keys appear only when they can be computed: `dino_consistency` always;
        `dino_similarity` and `dino_consistency_ref` with a reference; `clip_score` with a prompt, plus
        `clip_score_ref` and `clip_delta` when both are given."""
        picks = sample_indices(len(frames), self.frames_sampled)
        own = self.dino_embed([frames[i] for i in picks])
        result: dict[str, Any] = {"dino_consistency": consecutive_consistency(own)}
        if self.keep_descriptor:
            result["descriptor"] = self.descriptor(own)

        reference_sample = None
        if reference_frames is not None:
            ref_picks = sample_indices(len(reference_frames), self.frames_sampled)
            reference_sample = [reference_frames[i] for i in ref_picks]
            reference_embeddings = self.dino_embed(reference_sample)
            if len(reference_embeddings) == len(own):
                result["dino_similarity"] = float(cosine_rows(own, reference_embeddings).mean())
            result["dino_consistency_ref"] = consecutive_consistency(reference_embeddings)
            if self.keep_descriptor:
                result["descriptor_ref"] = self.descriptor(reference_embeddings)

        if prompt:
            text = self.clip_text_embed(prompt)
            result["clip_score"] = float(cosine_rows(self.clip_image_embed([frames[i] for i in picks]), text).mean())
            if reference_sample is not None:
                result["clip_score_ref"] = float(cosine_rows(self.clip_image_embed(reference_sample), text).mean())
                result["clip_delta"] = result["clip_score"] - result["clip_score_ref"]
        return result
