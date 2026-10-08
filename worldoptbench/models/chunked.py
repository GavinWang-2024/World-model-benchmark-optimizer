"""Chunked rollouts: run an imagination rollout as observe -> K-step chunks -> decode, each a small
tensor function run through `tensor_executor`.

Why (DESIGN_STEP_HOOK.md): a CUDA graph for the whole rollout is captured per horizon and Python cannot
act while it runs. Chunks are captured once per chunk size and reused for any horizon (the last chunk is
padded and trimmed), and the host regains control between chunks, which is where early stopping,
re-anchoring and streaming will hook in. The cost is one executor call per chunk (input copies, a replay
and an output clone) instead of one for the whole rollout.

Equivalence: the noise for all steps is drawn once, after the observe step and before any chunk, which is
the order the single-function rollout consumes the random stream in, so a chunked rollout with the same
seed matches the unchunked one whatever the chunk size. That only works for imagination backends whose
noise is an explicit input (`draw_noise`); the repo's own sampler draws inside each step and cannot be
chunked without changing its random stream.

This module is pure tensor plumbing (no Dreamer imports); the model supplies the three functions.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any


def run_chunked(
    executor: Callable[..., Any],
    observe_fn: Callable[..., Any],
    chunk_fn: Callable[..., Any],
    decode_fn: Callable[..., Any],
    noise_fn: Callable[[Any, int], Any],
    image: Any,
    prev_actions: Any,
    future_actions: Any,
    chunk_steps: int,
) -> Any:
    """Runs the rollout in chunks and returns the decoded frames for exactly N = future_actions.shape[1] steps.

    The functions (each takes and returns single tensors; build them once so executors can cache on identity):
      observe_fn(image, prev_actions) -> state, (B, state_dim): the packed latent after the context frames.
      chunk_fn(state, actions, noise) -> (B, state_dim + K * feature_dim): the packed next state, then the
          K feature vectors of the chunk. actions is (B, K, A); noise is (K, B, ...) as `noise_fn` produced it.
      decode_fn(features) -> frames (B, K, ...): features is (B, K, feature_dim).
      noise_fn(state, steps) -> noise (steps, B, ...), called once, after observe_fn and before any chunk.
    """
    import torch

    if chunk_steps < 1:
        raise ValueError("chunk_steps must be >= 1")
    batch, steps, action_dim = future_actions.shape
    if steps < 1:
        raise ValueError("a rollout needs at least one step")

    state = executor(observe_fn, image, prev_actions)
    state_dim = state.shape[1]
    noise = noise_fn(state, steps)

    chunks = math.ceil(steps / chunk_steps)
    padding = chunks * chunk_steps - steps
    if padding:
        # the last chunk is padded to the common size (so one captured graph serves every horizon) and
        # trimmed below; the padded steps' results are never used
        future_actions = torch.cat([future_actions, future_actions.new_zeros(batch, padding, action_dim)], dim=1)
        noise = torch.cat([noise, noise.new_zeros(padding, *noise.shape[1:])], dim=0)

    frames = []
    for index in range(chunks):
        window = slice(index * chunk_steps, (index + 1) * chunk_steps)
        packed = executor(chunk_fn, state, future_actions[:, window].contiguous(), noise[window].contiguous())
        state = packed[:, :state_dim]
        features = packed[:, state_dim:].reshape(batch, chunk_steps, -1).contiguous()
        frames.append(executor(decode_fn, features))
    return torch.cat(frames, dim=1)[:, :steps]
