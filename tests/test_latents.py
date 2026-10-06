"""Latent-only rollouts (catalog C4): the shared imagination helper and the wiring of `generate_latents`. CPU, no Dreamer repo."""

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from worldoptbench.models.dreamer import DreamerWorldModel, _imagine_decode, _imagine_features  # noqa: E402


class FakeDyn:
    """img_step adds the action sum to deter; features are stoch and deter side by side."""

    def get_feat(self, state):
        stoch = state["stoch"]
        return torch.cat([stoch.reshape(stoch.shape[0], stoch.shape[1], -1), state["deter"]], dim=-1)

    def img_step(self, state, action):
        return {"stoch": state["stoch"], "deter": state["deter"] + action.sum(-1, keepdim=True)}

    def imagine_with_action(self, actions, state):
        stochs, deters = [], []
        for t in range(actions.shape[1]):
            state = self.img_step(state, actions[:, t])
            stochs.append(state["stoch"])
            deters.append(state["deter"])
        return {"stoch": torch.stack(stochs, 1), "deter": torch.stack(deters, 1)}


def _inputs(steps=4):
    stoch, deter = torch.zeros(1, 2, 3), torch.zeros(1, 1)
    actions = torch.arange(steps, dtype=torch.float32).reshape(1, steps, 1).repeat(1, 1, 2)  # step t: (t, t)
    return stoch, deter, actions


def test_features_are_the_latents_of_every_imagined_step_and_the_lean_loop_matches_the_repos_scan():
    wm = types.SimpleNamespace(dynamics=FakeDyn())
    stoch, deter, actions = _inputs()
    scan = _imagine_features(wm, stoch, deter, actions, lean=False)
    lean = _imagine_features(wm, stoch, deter, actions, lean=True)
    assert scan.shape == (1, 4, 2 * 3 + 1)
    assert torch.equal(scan, lean)
    # deter accumulates the action sum: after steps 0..3 with sums 0, 2, 4, 6 -> 0, 2, 6, 12
    assert scan[0, :, -1].tolist() == [0.0, 2.0, 6.0, 12.0]


def test_a_backend_replaces_the_loop_and_its_output_is_what_the_features_are_built_from():
    calls = []

    def backend(wm, stoch, deter, actions):
        calls.append(actions.shape)
        return {"stoch": torch.ones(1, 4, 2, 3), "deter": torch.full((1, 4, 1), 7.0)}

    wm = types.SimpleNamespace(dynamics=FakeDyn())
    stoch, deter, actions = _inputs()
    features = _imagine_features(wm, stoch, deter, actions, lean=False, backend=backend)
    assert calls == [(1, 4, 2)] and features.shape == (1, 4, 7) and float(features[0, 0, -1]) == 7.0


def test_decoding_the_features_gives_exactly_what_imagine_decode_returns():
    # _imagine_decode must be "features, then decode": refactoring it into _imagine_features changed nothing
    seen = {}
    decoded = types.SimpleNamespace(mode=lambda: torch.full((1, 4, 2, 2, 3), 0.5))

    def decoder(features):
        seen["features"] = features
        return {"image": decoded}

    wm = types.SimpleNamespace(dynamics=FakeDyn(), heads={"decoder": decoder})
    stoch, deter, actions = _inputs()
    frames = _imagine_decode(wm, stoch, deter, actions, lean=False)
    assert torch.equal(seen["features"], _imagine_features(wm, stoch, deter, actions, lean=False))
    assert frames.dtype == torch.uint8 and frames.shape == (1, 4, 2, 2, 3)


def _model():
    repo = types.SimpleNamespace(task="t", steps_per_second=20.0, config=types.SimpleNamespace(num_actions=2))
    model = DreamerWorldModel(repo, ".")
    model._load = lambda: object()
    return model


def test_generate_latents_prepares_like_generate_and_runs_the_decoder_free_path():
    model = _model()
    seen = {}

    def fake_run(contexts, prev_actions, futures, seed, latents=False):
        seen.update(latents=latents, seed=seed, n_context=len(contexts[0]), n_future=len(futures[0]))
        return np.arange(1 * 20 * 5, dtype=np.float32).reshape(1, 20, 5)

    model._run = fake_run
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)] * 5
    out = model.generate_latents(init_video=frames, actions=[np.zeros(2, np.float32)] * 20, horizon=1.0, seed=9)
    assert seen == {"latents": True, "seed": 9, "n_context": 5, "n_future": 20}
    assert out.shape == (20, 5)  # one feature vector per step, the batch dimension removed


def test_generate_latents_validates_the_horizon_like_generate_does():
    model = _model()
    frames = [np.zeros((4, 4, 3), dtype=np.uint8)] * 5
    with pytest.raises(ValueError, match="actions were given"):
        model.generate_latents(init_video=frames, actions=[np.zeros(2, np.float32)] * 3, horizon=1.0)
