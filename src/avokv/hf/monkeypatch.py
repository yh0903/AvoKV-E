"""Hugging Face hooks used by the paper's Llama and Qwen3 runners.

This integration is adapted from the Apache-2.0 R-KV Hugging Face runner and
has been modified for AvoKV-E. See ``NOTICE`` for attribution.
"""

from __future__ import annotations

import os

if os.environ.get("AVOKV_DISABLE_FLASH_ATTN", "").lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    import transformers.utils as transformers_utils
    from transformers.utils import import_utils

    def _flash_attn_disabled(*args, **kwargs):
        return False

    import_utils.is_flash_attn_2_available = _flash_attn_disabled
    import_utils.is_flash_attn_greater_or_equal = _flash_attn_disabled
    transformers_utils.is_flash_attn_2_available = _flash_attn_disabled
    transformers_utils.is_flash_attn_greater_or_equal = _flash_attn_disabled

from transformers.models.llama import modeling_llama

try:
    from transformers.models.qwen3 import modeling_qwen3
except ModuleNotFoundError:
    modeling_qwen3 = None

from .modeling import (
    CausalLM_forward,
    LlamaAttention_forward,
    LlamaAttention_init,
    Qwen3Attention_forward,
    Qwen3Attention_init,
)

_ORIGINALS = {}
for _family, _module in (("llama", modeling_llama), ("qwen3", modeling_qwen3)):
    if _module is not None:
        _attention = getattr(_module, "LlamaAttention" if _family == "llama" else "Qwen3Attention")
        _causal_lm = getattr(_module, "LlamaForCausalLM" if _family == "llama" else "Qwen3ForCausalLM")
        _ORIGINALS[_family] = (_attention, _attention.__init__, _attention.forward,
                              _causal_lm, _causal_lm.forward)


def restore_model_family(family: str) -> None:
    """Restore original class hooks before constructing a FullKV model."""
    attention, init, forward, causal_lm, lm_forward = _ORIGINALS[family]
    attention.__init__ = init
    attention.forward = forward
    causal_lm.forward = lm_forward


def replace_llama(compression_config):
    def init_wrapper(self, config, layer_idx):
        LlamaAttention_init(self, config, layer_idx, compression_config)

    modeling_llama.LlamaAttention.__init__ = init_wrapper
    modeling_llama.LlamaAttention.forward = LlamaAttention_forward
    modeling_llama.LlamaForCausalLM.forward = CausalLM_forward


def replace_qwen3(compression_config):
    if modeling_qwen3 is None:
        raise ImportError(
            "Qwen3 support requires a Transformers version with transformers.models.qwen3"
        )

    def init_wrapper(self, config, layer_idx):
        Qwen3Attention_init(self, config, layer_idx, compression_config)

    modeling_qwen3.Qwen3Attention.__init__ = init_wrapper
    modeling_qwen3.Qwen3Attention.forward = Qwen3Attention_forward
    modeling_qwen3.Qwen3ForCausalLM.forward = CausalLM_forward


def patch_model_family(model_name_or_path: str, compression_config: dict) -> str:
    """Patch the supported Transformers model family and return its name."""
    model_name = model_name_or_path.lower()
    if "qwen3" in model_name:
        replace_qwen3(compression_config)
        return "qwen3"
    if "llama" in model_name:
        replace_llama(compression_config)
        return "llama"
    raise ValueError(
        "AvoKV-E currently supports Llama-family and Qwen3 causal LMs; "
        f"could not infer a supported family from {model_name_or_path!r}"
    )


__all__ = ["patch_model_family", "replace_llama", "replace_qwen3", "restore_model_family"]
