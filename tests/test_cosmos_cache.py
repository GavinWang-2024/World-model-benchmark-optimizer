"""Making the Cosmos transformer cacheable (registry entry, CacheMixin, guidance-pass cache contexts) on a tiny CPU model."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers import CosmosTransformer3DModel
from diffusers.hooks import FirstBlockCacheConfig

from worldoptbench.models.cosmos_predict import make_cacheable, register_cosmos_blocks


def _tiny():
    torch.manual_seed(0)
    return CosmosTransformer3DModel(
        in_channels=17, out_channels=16, num_attention_heads=2, attention_head_dim=16, num_layers=3, mlp_ratio=2.0,
        text_embed_dim=64, adaln_lora_dim=16, max_size=(4, 32, 32), patch_size=(1, 2, 2), rope_scale=(1.0, 3.0, 3.0),
        concat_padding_mask=True, extra_pos_embed_type=None,
    ).eval()


def _inputs(text_seed):
    generator = torch.Generator().manual_seed(text_seed)
    return {
        "hidden_states": torch.randn(1, 16, 2, 8, 8, generator=torch.Generator().manual_seed(1)),
        "condition_mask": torch.zeros(1, 1, 2, 8, 8),
        "timestep": torch.full((1, 1, 2, 1, 1), 0.5),
        "encoder_hidden_states": torch.randn(1, 5, 64, generator=generator),
        "padding_mask": torch.zeros(1, 1, 8, 8),
        "return_dict": False,
    }


def test_registry_entry_is_added_once_and_the_class_gains_the_cache_api():
    register_cosmos_blocks()
    register_cosmos_blocks()  # idempotent
    model = make_cacheable(_tiny())
    assert hasattr(model, "enable_cache") and hasattr(model, "cache_context")
    assert make_cacheable(model) is model  # idempotent, same object


def test_cacheable_model_computes_exactly_what_the_stock_model_does():
    stock = _tiny()
    inputs = _inputs(2)
    with torch.no_grad():
        expected = stock(**inputs)[0]
        got = make_cacheable(stock)(**inputs)[0]
    assert torch.equal(expected, got)


def test_the_negative_prompts_embeddings_select_the_uncond_context_and_it_is_cleared_after(monkeypatch):
    from diffusers.hooks import HookRegistry

    model = make_cacheable(_tiny())
    seen = []
    original = HookRegistry._set_context
    monkeypatch.setattr(HookRegistry, "_set_context", lambda self, name=None: (seen.append(name), original(self, name))[1])
    positive, negative = _inputs(2), _inputs(3)
    model._uncond_embeds = negative["encoder_hidden_states"]
    with torch.no_grad():
        model(**positive)
        model(**negative)
        model(**positive)
    assert seen == ["cond", None, "uncond", None, "cond", None]


def test_a_diffusers_block_cache_can_now_be_enabled_and_a_zero_threshold_changes_nothing():
    stock = _tiny()
    inputs = _inputs(2)
    with torch.no_grad():
        expected = stock(**inputs)[0]
    model = make_cacheable(stock)
    model.enable_cache(FirstBlockCacheConfig(threshold=0.0))
    with torch.no_grad():
        first = model(**inputs)[0]
        second = model(**inputs)[0]
    assert torch.allclose(first, expected, atol=1e-6) and torch.allclose(second, expected, atol=1e-6)
    model.disable_cache()
