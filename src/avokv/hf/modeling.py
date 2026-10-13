"""Hugging Face model integration for AvoKV-E.

This module is adapted from the Apache-2.0 R-KV Hugging Face integration and
modified to use AvoKV-E's eviction policy. See NOTICE for full attribution.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import logging

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3RMSNorm
except ModuleNotFoundError:
    Qwen3RMSNorm = None
    Qwen3Config = None
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from avokv.compression import AvoKVE
from avokv.compression.policy import update_shared_cache

KV_COMPRESSION_MAP = {
    "avokv_e": AvoKVE,
}

logger = logging.get_logger(__name__)


def _past_cache_seq_length(past_key_values):
    if past_key_values is None:
        return 0
    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if callable(get_seq_length):
        try:
            return int(get_seq_length())
        except (TypeError, ValueError):
            pass
    try:
        if len(past_key_values) == 0:
            return 0
        first_layer = past_key_values[0]
        first_key = first_layer[0] if isinstance(first_layer, (tuple, list)) else first_layer
        if torch.is_tensor(first_key) and first_key.ndim >= 2:
            return int(first_key.shape[-2])
    except (IndexError, KeyError, TypeError):
        pass
    return -1


def _set_cache_layer(past_key_value, layer_idx, key_states, value_states):
    if hasattr(past_key_value, "key_cache") and hasattr(past_key_value, "value_cache"):
        past_key_value.key_cache[layer_idx] = key_states
        past_key_value.value_cache[layer_idx] = value_states
        return

    layers = getattr(past_key_value, "layers", None)
    if layers is not None and layer_idx < len(layers):
        layers[layer_idx].keys = key_states
        layers[layer_idx].values = value_states
        if hasattr(layers[layer_idx], "is_initialized"):
            layers[layer_idx].is_initialized = True
        return

    raise AttributeError("Unsupported cache layout for compressed KV replacement")


def _clear_kv_valid_mask(past_key_value, layer_idx):
    if past_key_value is None or not hasattr(past_key_value, "kv_valid_masks"):
        return
    past_key_value.kv_valid_masks.pop(layer_idx, None)


def _valid_mask_from_attention_mask(attention_mask, key_states):
    if attention_mask is None:
        return None

    kv_len = key_states.shape[-2]
    if attention_mask.ndim == 2:
        valid_mask = attention_mask.to(device=key_states.device, dtype=torch.bool)
    elif attention_mask.ndim >= 4:
        base_valid = attention_mask[:, 0, -1, :].to(device=key_states.device)
        if base_valid.dtype == torch.bool:
            valid_mask = base_valid
        else:
            valid_mask = base_valid == 0
    else:
        return None

    if valid_mask.shape[-1] < kv_len:
        pad = torch.zeros(
            valid_mask.shape[0],
            kv_len - valid_mask.shape[-1],
            device=key_states.device,
            dtype=torch.bool,
        )
        valid_mask = torch.cat([pad, valid_mask], dim=-1)
    elif valid_mask.shape[-1] > kv_len:
        valid_mask = valid_mask[:, -kv_len:]
    return valid_mask


def _seed_kv_valid_mask_from_attention_mask(
    past_key_value, layer_idx, kv_cluster, attention_mask, key_states
):
    if past_key_value is None:
        return
    valid_mask = _valid_mask_from_attention_mask(attention_mask, key_states)
    if valid_mask is None or bool(valid_mask.all().item()):
        _clear_kv_valid_mask(past_key_value, layer_idx)
        setattr(kv_cluster, "kv_valid_mask", None)
        return

    if not hasattr(past_key_value, "kv_valid_masks"):
        past_key_value.kv_valid_masks = {}
    past_key_value.kv_valid_masks[layer_idx] = valid_mask
    setattr(kv_cluster, "kv_valid_mask", valid_mask)


def _append_kv_valid_mask(past_key_value, layer_idx, key_states):
    if past_key_value is None or not hasattr(past_key_value, "kv_valid_masks"):
        return
    valid_mask = past_key_value.kv_valid_masks.get(layer_idx)
    if valid_mask is None:
        return

    kv_len = key_states.shape[-2]
    valid_mask = valid_mask.to(device=key_states.device, dtype=torch.bool)
    if valid_mask.shape[-1] < kv_len:
        added = torch.ones(
            valid_mask.shape[0],
            kv_len - valid_mask.shape[-1],
            device=key_states.device,
            dtype=torch.bool,
        )
        valid_mask = torch.cat([valid_mask, added], dim=-1)
    elif valid_mask.shape[-1] > kv_len:
        valid_mask = valid_mask[:, -kv_len:]
    past_key_value.kv_valid_masks[layer_idx] = valid_mask


def _store_kv_valid_mask(past_key_value, layer_idx, kv_cluster, key_states):
    if past_key_value is None:
        return

    valid_mask = getattr(kv_cluster, "last_kv_valid_mask", None)
    if valid_mask is None:
        _clear_kv_valid_mask(past_key_value, layer_idx)
        setattr(kv_cluster, "kv_valid_mask", None)
        return

    valid_mask = valid_mask.to(device=key_states.device, dtype=torch.bool)
    kv_len = key_states.shape[-2]
    if valid_mask.shape[-1] < kv_len:
        added = torch.ones(
            valid_mask.shape[0],
            kv_len - valid_mask.shape[-1],
            device=key_states.device,
            dtype=torch.bool,
        )
        valid_mask = torch.cat([valid_mask, added], dim=-1)
    elif valid_mask.shape[-1] > kv_len:
        valid_mask = valid_mask[:, -kv_len:]

    if bool(valid_mask.all().item()):
        _clear_kv_valid_mask(past_key_value, layer_idx)
        setattr(kv_cluster, "kv_valid_mask", None)
        return

    if not hasattr(past_key_value, "kv_valid_masks"):
        past_key_value.kv_valid_masks = {}
    past_key_value.kv_valid_masks[layer_idx] = valid_mask
    setattr(kv_cluster, "kv_valid_mask", valid_mask)


def _sync_cluster_kv_valid_mask(past_key_value, layer_idx, kv_cluster, key_states):
    if past_key_value is None or not hasattr(past_key_value, "kv_valid_masks"):
        setattr(kv_cluster, "kv_valid_mask", None)
        return

    valid_mask = past_key_value.kv_valid_masks.get(layer_idx)
    if valid_mask is None:
        setattr(kv_cluster, "kv_valid_mask", None)
        return

    valid_mask = valid_mask.to(device=key_states.device, dtype=torch.bool)
    kv_len = key_states.shape[-2]
    if valid_mask.shape[-1] < kv_len:
        added = torch.ones(
            valid_mask.shape[0],
            kv_len - valid_mask.shape[-1],
            device=key_states.device,
            dtype=torch.bool,
        )
        valid_mask = torch.cat([valid_mask, added], dim=-1)
    elif valid_mask.shape[-1] > kv_len:
        valid_mask = valid_mask[:, -kv_len:]
    past_key_value.kv_valid_masks[layer_idx] = valid_mask
    setattr(kv_cluster, "kv_valid_mask", valid_mask)


def _get_kv_valid_mask(past_key_value, layer_idx):
    if past_key_value is None or not hasattr(past_key_value, "kv_valid_masks"):
        return None
    return past_key_value.kv_valid_masks.get(layer_idx)


def _apply_kv_valid_mask(
    attention_mask, valid_mask, query_states, key_states, attn_implementation=None
):
    if valid_mask is None:
        return attention_mask

    valid_mask = valid_mask.to(device=key_states.device, dtype=torch.bool)
    kv_len = key_states.shape[-2]
    if valid_mask.shape[-1] < kv_len:
        added = torch.ones(
            valid_mask.shape[0],
            kv_len - valid_mask.shape[-1],
            device=key_states.device,
            dtype=torch.bool,
        )
        valid_mask = torch.cat([valid_mask, added], dim=-1)
    elif valid_mask.shape[-1] > kv_len:
        valid_mask = valid_mask[:, -kv_len:]

    if bool(valid_mask.all().item()):
        return attention_mask

    if attn_implementation == "flash_attention_2":
        if attention_mask is None:
            return valid_mask.to(dtype=torch.int32)
        if attention_mask.ndim == 2:
            base_mask = attention_mask.to(device=key_states.device)
            if base_mask.shape[-1] < kv_len:
                pad = torch.ones(
                    base_mask.shape[0],
                    kv_len - base_mask.shape[-1],
                    device=base_mask.device,
                    dtype=base_mask.dtype,
                )
                base_mask = torch.cat([pad, base_mask], dim=-1)
            elif base_mask.shape[-1] > kv_len:
                base_mask = base_mask[:, -kv_len:]
            combined = base_mask.to(dtype=torch.bool) & valid_mask
            if attention_mask.dtype == torch.bool:
                return combined
            return combined.to(dtype=attention_mask.dtype)
        if attention_mask.ndim >= 4:
            base_valid = attention_mask[:, 0, -1, :].to(device=key_states.device)
            if base_valid.shape[-1] < kv_len:
                pad = torch.zeros(
                    base_valid.shape[0],
                    kv_len - base_valid.shape[-1],
                    device=base_valid.device,
                    dtype=base_valid.dtype,
                )
                base_valid = torch.cat([pad, base_valid], dim=-1)
            elif base_valid.shape[-1] > kv_len:
                base_valid = base_valid[:, -kv_len:]
            if base_valid.dtype == torch.bool:
                combined = base_valid & valid_mask
            else:
                combined = (base_valid == 0) & valid_mask
            return combined.to(dtype=torch.int32)

    q_len = query_states.shape[-2]
    invalid = ~valid_mask[:, None, None, :]
    min_value = torch.finfo(query_states.dtype).min
    additive = torch.zeros(
        valid_mask.shape[0],
        1,
        q_len,
        kv_len,
        device=query_states.device,
        dtype=query_states.dtype,
    )
    additive = additive.masked_fill(invalid, min_value)

    if attention_mask is None:
        return additive

    base_mask = attention_mask
    if base_mask.shape[-1] < kv_len:
        pad = torch.zeros(
            *base_mask.shape[:-1],
            kv_len - base_mask.shape[-1],
            device=base_mask.device,
            dtype=base_mask.dtype,
        )
        base_mask = torch.cat([pad, base_mask], dim=-1)
    elif base_mask.shape[-1] > kv_len:
        base_mask = base_mask[..., -kv_len:]

    if base_mask.shape[-2] < q_len:
        pad = torch.zeros(
            *base_mask.shape[:-2],
            q_len - base_mask.shape[-2],
            base_mask.shape[-1],
            device=base_mask.device,
            dtype=base_mask.dtype,
        )
        base_mask = torch.cat([pad, base_mask], dim=-2)
    elif base_mask.shape[-2] > q_len:
        base_mask = base_mask[..., -q_len:, :]

    return base_mask.to(dtype=query_states.dtype) + additive


def LlamaAttention_init(self, config: LlamaConfig, layer_idx: int, compression_config: dict):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True

    self.q_proj = nn.Linear(
        config.hidden_size,
        config.num_attention_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.k_proj = nn.Linear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.v_proj = nn.Linear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.o_proj = nn.Linear(
        config.num_attention_heads * self.head_dim,
        config.hidden_size,
        bias=config.attention_bias,
    )

    # =============== New logic start ===============
    self.config.update(compression_config)
    self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
        **compression_config["method_config"]
    )
    self.kv_cluster.layer_idx = layer_idx
    # =============== New logic end =================


def LlamaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_value: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
    if past_key_value is None:
        past_key_value = kwargs.pop("past_key_values", None)
    else:
        kwargs.pop("past_key_values", None)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # =============== Enable Query Cache ============
        if not hasattr(past_key_value, "query_cache"):
            past_key_value.query_cache = {}

        if self.layer_idx not in past_key_value.query_cache:
            # prefill stage
            bsz, n_heads, _, head_dim = query_states.shape
            past_key_value.query_cache[self.layer_idx] = torch.empty(bsz, n_heads, 0, head_dim)
            past_key_value.query_cache[self.layer_idx] = query_states[
                :, :, -self.config.method_config["window_size"] :, :
            ]
            _seed_kv_valid_mask_from_attention_mask(
                past_key_value,
                self.layer_idx,
                self.kv_cluster,
                attention_mask,
                key_states,
            )
        else:
            # Add current query to cache
            past_key_value.query_cache[self.layer_idx] = torch.cat(
                (past_key_value.query_cache[self.layer_idx], query_states), dim=2
            )  # [batch, n_q_heads, seq_len, head_dim]

            # Keep only window_size most recent queries
            window_size = self.config.method_config["window_size"]
            if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
                past_key_value.query_cache[self.layer_idx] = past_key_value.query_cache[
                    self.layer_idx
                ][:, :, -window_size:, :]
        # =============== Enable Query Cache end =========

        # =============== decoding-time compression start ===============
        cached_queries = past_key_value.query_cache[self.layer_idx]
        attention_kv_valid_mask = None
        if self.config.compression is None:
            _sync_cluster_kv_valid_mask(past_key_value, self.layer_idx, self.kv_cluster, key_states)
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,  # Use cached queries instead of current query
                value_states,
            )

            if self.config.update_kv is True:
                past_key_value.update(
                    key_states_compress,
                    value_states_compress,
                    self.layer_idx,
                    cache_kwargs,
                )
                _store_kv_valid_mask(
                    past_key_value,
                    self.layer_idx,
                    self.kv_cluster,
                    key_states_compress,
                )
            else:
                past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )
                _clear_kv_valid_mask(past_key_value, self.layer_idx)

        elif self.config.compression is True:
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )
            _append_kv_valid_mask(past_key_value, self.layer_idx, key_states)
            _sync_cluster_kv_valid_mask(past_key_value, self.layer_idx, self.kv_cluster, key_states)
            attention_kv_valid_mask = _get_kv_valid_mask(past_key_value, self.layer_idx)

            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,  # Use cached queries instead of current query
                value_states,
            )

            if self.config.update_kv is True:
                _set_cache_layer(
                    past_key_value,
                    self.layer_idx,
                    key_states_compress,
                    value_states_compress,
                )
                _store_kv_valid_mask(
                    past_key_value,
                    self.layer_idx,
                    self.kv_cluster,
                    key_states_compress,
                )
            else:
                _clear_kv_valid_mask(past_key_value, self.layer_idx)
        else:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
            _append_kv_valid_mask(past_key_value, self.layer_idx, key_states)
            attention_kv_valid_mask = _get_kv_valid_mask(past_key_value, self.layer_idx)
        # =============== decoding-time compression end ===============

        attention_mask = _apply_kv_valid_mask(
            attention_mask,
            attention_kv_valid_mask,
            query_states,
            key_states,
            self.config._attn_implementation,
        )

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def Qwen3Attention_init(self, config: Qwen3Config, layer_idx: int, compression_config: dict):
    if Qwen3RMSNorm is None:
        raise ImportError(
            "Qwen3 support requires a Transformers version that provides transformers.models.qwen3"
        )

    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True

    self.q_proj = nn.Linear(
        config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
    )
    self.k_proj = nn.Linear(
        config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
    )
    self.v_proj = nn.Linear(
        config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
    )
    self.o_proj = nn.Linear(
        config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
    )
    self.q_norm = Qwen3RMSNorm(
        self.head_dim, eps=config.rms_norm_eps
    )  # unlike olmo, only on the head dim!
    self.k_norm = Qwen3RMSNorm(
        self.head_dim, eps=config.rms_norm_eps
    )  # thus post q_norm does not need reshape
    self.sliding_window = config.sliding_window
    if not (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        self.sliding_window = None

    # =============== New logic start ===============
    self.config.update(compression_config)
    self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
        **compression_config["method_config"]
    )
    self.kv_cluster.layer_idx = layer_idx
    # =============== New logic end =================


def Qwen3Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_value: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
    if past_key_value is None:
        past_key_value = kwargs.pop("past_key_values", None)
    else:
        kwargs.pop("past_key_values", None)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # =============== Enable Query Cache ============
        if not hasattr(past_key_value, "query_cache"):
            past_key_value.query_cache = {}

        if self.layer_idx not in past_key_value.query_cache:
            # prefill stage
            bsz, n_heads, _, head_dim = query_states.shape
            past_key_value.query_cache[self.layer_idx] = torch.empty(bsz, n_heads, 0, head_dim)
            past_key_value.query_cache[self.layer_idx] = query_states[
                :, :, -self.config.method_config["window_size"] :, :
            ]
            _seed_kv_valid_mask_from_attention_mask(
                past_key_value,
                self.layer_idx,
                self.kv_cluster,
                attention_mask,
                key_states,
            )
        else:
            # Add current query to cache
            past_key_value.query_cache[self.layer_idx] = torch.cat(
                (past_key_value.query_cache[self.layer_idx], query_states), dim=2
            )  # [batch, n_q_heads, seq_len, head_dim]

            # Keep only window_size most recent queries
            window_size = self.config.method_config["window_size"]
            if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
                past_key_value.query_cache[self.layer_idx] = past_key_value.query_cache[
                    self.layer_idx
                ][:, :, -window_size:, :]
        # =============== Enable Query Cache end =========

        # =============== decoding-time compression start ===============
        cached_queries = past_key_value.query_cache[self.layer_idx]
        attention_kv_valid_mask = None
        if self.config.compression is None:
            _sync_cluster_kv_valid_mask(past_key_value, self.layer_idx, self.kv_cluster, key_states)
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,  # Use cached queries instead of current query
                value_states,
            )

            past_key_value.update(
                key_states_compress,
                value_states_compress,
                self.layer_idx,
                cache_kwargs,
            )
            _store_kv_valid_mask(
                past_key_value,
                self.layer_idx,
                self.kv_cluster,
                key_states_compress,
            )

        elif self.config.compression is True:
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )
            _append_kv_valid_mask(past_key_value, self.layer_idx, key_states)
            _sync_cluster_kv_valid_mask(past_key_value, self.layer_idx, self.kv_cluster, key_states)
            attention_kv_valid_mask = _get_kv_valid_mask(past_key_value, self.layer_idx)

            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,  # Use cached queries instead of current query
                value_states,
            )

            _set_cache_layer(
                past_key_value,
                self.layer_idx,
                key_states_compress,
                value_states_compress,
            )
            _store_kv_valid_mask(
                past_key_value,
                self.layer_idx,
                self.kv_cluster,
                key_states_compress,
            )
        else:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
            _append_kv_valid_mask(past_key_value, self.layer_idx, key_states)
            attention_kv_valid_mask = _get_kv_valid_mask(past_key_value, self.layer_idx)
        # =============== decoding-time compression end ===============

        attention_mask = _apply_kv_valid_mask(
            attention_mask,
            attention_kv_valid_mask,
            query_states,
            key_states,
            self.config._attn_implementation,
        )

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,  # diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def CausalLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | list[torch.FloatTensor] | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs,
) -> tuple | CausalLMOutputWithPast:
    output_attentions = (
        output_attentions if output_attentions is not None else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    batch_invariant_compression = bool(getattr(self.config, "batch_invariant_compression", False))
    current_seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
    is_prefill = _past_cache_seq_length(past_key_values) == 0
    policies = [layer.self_attn.kv_cluster for layer in self.model.layers]
    shared = policies[0].config.selection_scope == "shared"
    batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
    if batch_size != 1:
        raise ValueError("AvoKV-E generation requires batch_size=1")
    if use_cache is False:
        raise ValueError("the eviction integration requires use_cache=True")
    if is_prefill and attention_mask is not None and attention_mask.ndim == 2:
        if not bool(attention_mask.all()):
            raise ValueError("the release integration requires an unpadded prompt")
    if not is_prefill and current_seq_len != 1:
        raise ValueError("compressed decoding supports one new token per forward")
    compression_due = False

    if is_prefill:
        self.length = current_seq_len
        self.generated_length = 0
        if self.config.compression_content == "think":
            self.after_think = False
        budget_scope = self.config.method_config.get("budget_scope", "total")
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        if budget_scope == "decode" and batch_size != 1:
            raise ValueError(
                "budget_scope='decode' currently requires eval_batch_size=1 "
                "because each prompt has a different total KV capacity."
            )
        for layer in self.model.layers:
            layer.self_attn.config.compression = False
            reset_for_prefill = getattr(layer.self_attn.kv_cluster, "reset_for_prefill", None)
            if reset_for_prefill is not None:
                reset_for_prefill(current_seq_len)
    else:
        if not hasattr(past_key_values, "avokv_seen_tokens"):
            raise ValueError("cannot resume a cache created outside the AvoKV-E integration")
        self.length = past_key_values.avokv_seen_tokens + current_seq_len
        self.generated_length = past_key_values.avokv_generated_tokens + current_seq_len
        for policy in policies:
            policy.prefill_tokens = past_key_values.avokv_prefill_tokens
        if self.config.divide_method == "step_length":
            compression_due = (
                self.generated_length > 0 and self.generated_length % self.config.divide_length == 0
            )
            if self.config.compression_content == "think" and not getattr(
                self, "after_think", False
            ):
                compression_due = False
            for layer in self.model.layers:
                layer.self_attn.config.compression = compression_due and not shared

    # DynamicCache.get_seq_length() reflects the compressed cache length. Use
    # the logical sequence length for RoPE positions so eviction cannot rewind
    # position IDs in Transformers versions that infer them from the cache.
    if position_ids is None:
        first_position = self.length - current_seq_len
        position_ids = torch.arange(
            first_position,
            self.length,
            device=input_ids.device if input_ids is not None else inputs_embeds.device,
            dtype=torch.long,
        ).unsqueeze(0)
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        position_ids = position_ids.expand(batch_size, -1)

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    cache = outputs.past_key_values
    if cache is not None:
        cache.avokv_seen_tokens = self.length
        cache.avokv_generated_tokens = self.generated_length
        cache.avokv_prefill_tokens = current_seq_len if is_prefill else policies[0].prefill_tokens
        if shared:
            new_positions = position_ids[0].detach()
            positions = new_positions if is_prefill else torch.cat(
                (cache.avokv_positions.to(new_positions.device), new_positions)
            )
            if compression_due:
                states = [
                    (cache.layers[i].keys, cache.query_cache[i], cache.layers[i].values)
                    for i in range(len(policies))
                ]
                compressed, positions = update_shared_cache(policies, states, positions)
                for i, (keys, values) in enumerate(compressed):
                    _set_cache_layer(cache, i, keys, values)
            cache.avokv_positions = positions

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = (
        slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    )
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    # =============== Step-level Compression logic start ===============
    # assume non-batch input, shape: [1, logits_to_keep, vocab_size]
    predicted_token_ids = logits[:, -1, :].argmax(dim=-1)

    if self.config.compression_content == "think" and not self.after_think:
        if batch_invariant_compression and predicted_token_ids.numel() > 1:
            raise ValueError(
                "batch_invariant_compression with batch_size > 1 only supports "
                "compression_content='all'; content-triggered think boundaries need "
                "per-sequence ragged cache state."
            )
        self.after_think = predicted_token_ids[0].cpu().item() in self.after_think_token_ids

    if self.config.divide_method == "newline":
        if batch_invariant_compression and predicted_token_ids.numel() > 1:
            raise ValueError(
                "batch_invariant_compression with batch_size > 1 only supports "
                "divide_method='step_length'; newline-triggered compression needs "
                "per-sequence ragged cache state."
            )
        is_newline = predicted_token_ids[0].cpu().item() in self.newline_token_ids
    elif self.config.divide_method == "step_length":
        is_newline = (
            self.generated_length > 0 and self.generated_length % self.config.divide_length == 0
        )
    else:
        raise ValueError(f"Invalid divide_method: {self.config.divide_method}")

    if self.config.compression_content == "think" and self.after_think:
        is_newline = False

    # Set compression flag for all layers at once
    for layer in self.model.layers:
        layer.self_attn.config.compression = is_newline and not shared
    # =============== Step-level Compression logic end =================

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.vocab_size,
            **kwargs,
        )

    if not return_dict:
        output = (logits,) + outputs.to_tuple()[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
