from __future__ import annotations

import pytest
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from avokv.compression import enable_metrics, reset_metrics, summarize_metrics
from avokv.hf import patch_model_family
from avokv.hf.monkeypatch import restore_model_family


@pytest.fixture(autouse=True)
def restore_hf_hooks():
    for family in ("llama", "qwen3"):
        restore_model_family(family)
    yield
    for family in ("llama", "qwen3"):
        restore_model_family(family)


def compression_config() -> dict:
    return {
        "method": "avokv_e",
        "method_config": {
            "budget": 4,
            "budget_scope": "decode",
            "window_size": 2,
            "observation_window": 2,
            "delay_window": 2,
            "lambda_p": 0.1,
            "lambda_r": 0.9,
            "lambda_b": 0.05,
            "first_tokens": 1,
            "redundancy_chunk_size": 2,
        },
        "compression": None,
        "update_kv": True,
    }


def configure_runtime(model) -> None:
    model.config.update(
        {
            "divide_method": "step_length",
            "divide_length": 2,
            "compression_content": "all",
            "batch_invariant_compression": False,
        }
    )
    model.newline_token_ids = []
    model.after_think_token_ids = []


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_tiny_hf_generation_triggers_eviction(family: str) -> None:
    torch.manual_seed(13)
    patch_model_family(family, compression_config())

    if family == "llama":
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=None,
            pad_token_id=0,
        )
        model = LlamaForCausalLM(config)
    else:
        config = Qwen3Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            bos_token_id=1,
            eos_token_id=None,
            pad_token_id=0,
        )
        model = Qwen3ForCausalLM(config)

    model.eval()
    configure_runtime(model)
    enable_metrics(True)
    reset_metrics()

    observed_positions = []

    def record_positions(_module, _args, kwargs):
        position_ids = kwargs.get("position_ids")
        if position_ids is None:
            position_ids = _args[1]
        observed_positions.extend(position_ids.flatten().tolist())

    position_hook = model.model.rotary_emb.register_forward_pre_hook(
        record_positions,
        with_kwargs=True,
    )

    try:
        generated = model.generate(
            input_ids=torch.tensor([[1, 7, 9]]),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            max_new_tokens=10,
            do_sample=False,
            use_cache=True,
        )
    finally:
        position_hook.remove()

    summary = summarize_metrics()
    assert generated.shape == (1, 13)
    assert summary["compression_events"] > 0
    assert summary["max_kv_len_after"] <= 7
    assert observed_positions == list(range(len(observed_positions)))


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_no_eviction_matches_fullkv_and_cache_is_sequence_local(family: str) -> None:
    model_class = LlamaForCausalLM if family == "llama" else Qwen3ForCausalLM
    config_class = LlamaConfig if family == "llama" else Qwen3Config
    config = config_class(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        eos_token_id=None,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(13)
    reference = model_class(config).eval()
    weights = reference.state_dict()
    inputs = torch.tensor([[1, 7, 9]])
    with torch.no_grad():
        expected = reference(input_ids=inputs, use_cache=True)
        expected_decode = reference(
            input_ids=torch.tensor([[3]]), past_key_values=expected.past_key_values, use_cache=True
        )
    settings = compression_config()
    settings["method_config"]["budget"] = 64
    patch_model_family(family, settings)
    model = model_class(config).eval()
    model.load_state_dict(weights)
    configure_runtime(model)
    with torch.no_grad():
        actual = model(input_ids=inputs, use_cache=True)
        torch.testing.assert_close(actual.logits, expected.logits)
        resumed = model(
            input_ids=torch.tensor([[3]]), past_key_values=actual.past_key_values, use_cache=True
        )
        torch.testing.assert_close(resumed.logits, expected_decode.logits)
        # Interleave an independent sequence, then resume the first cache.
        model(input_ids=torch.tensor([[2, 4]]), use_cache=True)
        resumed = model(
            input_ids=torch.tensor([[5]]), past_key_values=resumed.past_key_values, use_cache=True
        )
    assert resumed.past_key_values.avokv_positions.tolist() == [0, 1, 2, 3, 4]
    assert resumed.past_key_values.avokv_prefill_tokens == 3
    assert all(layer.self_attn.kv_cluster.prefill_tokens == 3 for layer in model.model.layers)


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_real_forward_keeps_one_shared_set_and_absolute_neighborhoods(family: str) -> None:
    model_class = LlamaForCausalLM if family == "llama" else Qwen3ForCausalLM
    config_class = LlamaConfig if family == "llama" else Qwen3Config
    config = config_class(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    patch_model_family(family, compression_config())
    model = model_class(config).eval()
    configure_runtime(model)
    with torch.no_grad():
        result = model(input_ids=torch.tensor([[1, 7, 9]]), use_cache=True)
        for step in range(12):
            result = model(
                input_ids=torch.tensor([[step + 1]]),
                past_key_values=result.past_key_values,
                use_cache=True,
            )
            cache = result.past_key_values
            assert cache.avokv_positions[-1].item() == step + 3
            assert cache.avokv_positions[:3].tolist() == [0, 1, 2]
            assert torch.all(cache.avokv_positions.diff() > 0)
            assert all(layer.keys.shape[-2] == len(cache.avokv_positions) for layer in cache.layers)
            if (step + 1) % 2 == 0:
                assert len(cache.avokv_positions) <= 7
                keeps = [
                    layer.self_attn.kv_cluster.last_keep_indices for layer in model.model.layers
                ]
                if keeps[0] is not None:
                    assert all(torch.equal(keeps[0], keep) for keep in keeps[1:])


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_explicit_per_head_compatibility_mode(family: str) -> None:
    model_class = LlamaForCausalLM if family == "llama" else Qwen3ForCausalLM
    config_class = LlamaConfig if family == "llama" else Qwen3Config
    config = config_class(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    settings = compression_config()
    settings["method_config"].update(
        {"selection_scope": "per_head", "redundancy_neighborhood": "global"}
    )
    patch_model_family(family, settings)
    model = model_class(config).eval()
    configure_runtime(model)
    with torch.no_grad():
        result = model(input_ids=torch.tensor([[1, 7, 9]]), use_cache=True)
        for step in range(8):
            result = model(
                input_ids=torch.tensor([[step + 1]]),
                past_key_values=result.past_key_values,
                use_cache=True,
            )
    assert result.past_key_values.avokv_seen_tokens == 11
    for layer in model.model.layers:
        assert layer.self_attn.kv_cluster.last_keep_indices.shape == (1, 2, 5)
    assert all(layer.keys.shape[-2] == 7 for layer in result.past_key_values.layers)


@pytest.mark.parametrize("invalid", ["batch", "padding"])
def test_unsupported_prompt_shapes_fail_explicitly(invalid: str) -> None:
    patch_model_family("llama", compression_config())
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(config).eval()
    configure_runtime(model)
    inputs = torch.tensor([[1, 7, 9]])
    mask = torch.tensor([[0, 1, 1]])
    if invalid == "batch":
        inputs, mask = inputs.expand(2, -1), torch.ones(2, 3)
    with pytest.raises(ValueError, match="batch_size=1|unpadded"):
        model(input_ids=inputs, attention_mask=mask, use_cache=True)
