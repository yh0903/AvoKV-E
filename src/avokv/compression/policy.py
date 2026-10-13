from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .metrics import record_kv_event


def percentile_rank(values: torch.Tensor) -> torch.Tensor:
    """Return ascending percentile ranks with averaged ranks for exact ties."""
    if values.ndim < 1:
        raise ValueError("values must have at least one dimension")

    size = values.shape[-1]
    if size == 0:
        return values.to(dtype=torch.float32)

    sorted_values, order = values.float().sort(dim=-1, stable=True)
    flat_values = sorted_values.reshape(-1, size)
    flat_order = order.reshape(-1, size)

    group_start = torch.ones_like(flat_values, dtype=torch.bool)
    if size > 1:
        group_start[:, 1:] = flat_values[:, 1:] != flat_values[:, :-1]
    group_ids = group_start.cumsum(dim=-1) - 1

    positions = torch.arange(size, device=values.device, dtype=torch.float32).expand(
        flat_values.shape[0], -1
    )
    rank_sums = torch.zeros_like(positions).scatter_add_(1, group_ids, positions)
    counts = torch.zeros_like(positions).scatter_add_(1, group_ids, torch.ones_like(positions))
    average_sorted_ranks = (rank_sums / counts.clamp_min(1.0)).gather(1, group_ids)

    flat_ranks = torch.empty_like(average_sorted_ranks)
    flat_ranks.scatter_(1, flat_order, average_sorted_ranks)
    return (flat_ranks / max(size - 1, 1)).reshape(values.shape)


def causal_attention_probabilities(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct causal attention for recent queries, grouped by KV head.

    Args:
        query_states: RoPE-transformed queries with shape ``[B, Hq, W, D]``.
        key_states: RoPE-transformed keys with shape ``[B, Hkv, T, D]``.

    Returns:
        Attention probabilities with shape ``[B, Hkv, G, W, T]``, where
        ``G = Hq / Hkv``. Queries are assumed to be the final ``W`` positions
        in the current cache. Each query is masked against keys created later.
    """
    if query_states.ndim != 4 or key_states.ndim != 4:
        raise ValueError("query_states and key_states must be rank-4 tensors")

    batch_size, query_heads, query_len, head_dim = query_states.shape
    key_batch, kv_heads, key_len, key_dim = key_states.shape
    if batch_size != key_batch or head_dim != key_dim:
        raise ValueError("query/key batch and head dimensions must match")
    if query_heads % kv_heads != 0:
        raise ValueError(f"query_heads={query_heads} must be divisible by kv_heads={kv_heads}")
    if query_len > key_len:
        raise ValueError("recent query window cannot be longer than the key cache")

    groups = query_heads // kv_heads
    grouped_queries = query_states.reshape(batch_size, kv_heads, groups, query_len, head_dim)
    logits = torch.matmul(
        grouped_queries.float(),
        key_states.float().unsqueeze(2).transpose(-1, -2),
    ) / math.sqrt(head_dim)

    query_positions = torch.arange(key_len - query_len, key_len, device=key_states.device)
    key_positions = torch.arange(key_len, device=key_states.device)
    causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    logits = logits.masked_fill(
        ~causal_mask.view(1, 1, 1, query_len, key_len),
        float("-inf"),
    )
    return F.softmax(logits, dim=-1, dtype=torch.float32)


def max_cosine_redundancy(
    key_states: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Maximum non-self cosine similarity for each candidate key.

    The computation is chunked over candidate rows so peak temporary memory is
    ``O(chunk_size * sequence_length)`` rather than ``O(sequence_length**2)``.
    """
    if key_states.ndim != 4:
        raise ValueError("key_states must have shape [B, Hkv, T, D]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    candidate_count = key_states.shape[-2]
    if candidate_count <= 1:
        return torch.zeros(
            key_states.shape[:-1],
            device=key_states.device,
            dtype=torch.float32,
        )

    normalized = F.normalize(key_states.float(), dim=-1, eps=1e-8)
    all_keys = normalized.transpose(-1, -2)
    outputs = []
    for start in range(0, candidate_count, chunk_size):
        end = min(start + chunk_size, candidate_count)
        similarities = torch.matmul(normalized[..., start:end, :], all_keys)
        local_rows = torch.arange(end - start, device=key_states.device)
        diagonal_columns = torch.arange(start, end, device=key_states.device)
        similarities[..., local_rows, diagonal_columns] = float("-inf")
        outputs.append(similarities.amax(dim=-1))
    return torch.cat(outputs, dim=-1)


def local_cosine_redundancy(
    key_states: torch.Tensor,
    positions: torch.Tensor | None = None,
    *,
    radius: int = 8,
) -> torch.Tensor:
    """Maximum non-self cosine within an absolute-token-position radius.

    A candidate with no surviving neighbor receives zero redundancy. Gaps
    created by earlier evictions do not turn distant tokens into neighbors.
    """
    if key_states.ndim != 4 or radius <= 0:
        raise ValueError("expected rank-4 keys and a positive neighborhood radius")
    count = key_states.shape[-2]
    if positions is None:
        positions = torch.arange(count, device=key_states.device)
    positions = positions.to(device=key_states.device)
    if positions.shape != (count,):
        raise ValueError("positions must contain one absolute position per candidate")
    normalized = F.normalize(key_states.float(), dim=-1, eps=1e-8)
    result = torch.full(key_states.shape[:-1], float("-inf"), device=key_states.device)
    if count == 0:
        return result
    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue
        targets = positions + offset
        indices = torch.searchsorted(positions, targets).clamp_max(count - 1)
        valid = positions[indices] == targets
        similarity = (normalized * normalized.index_select(-2, indices)).sum(-1)
        result = torch.maximum(result, similarity.masked_fill(~valid, float("-inf")))
    return torch.where(torch.isfinite(result), result, 0.0)


@dataclass(frozen=True)
class AvoKVEConfig:
    """Configuration used by the paper's eviction-only policy."""

    budget: int
    budget_scope: str = "decode"
    observation_window: int = 8
    delay_window: int = 8
    lambda_p: float = 0.1
    lambda_r: float = 0.9
    lambda_b: float = 0.05
    first_tokens: int = 4
    redundancy_chunk_size: int = 256
    selection_scope: str = "shared"
    redundancy_neighborhood: str = "local"
    redundancy_radius: int = 8

    def __post_init__(self) -> None:
        if self.budget_scope not in {"decode", "total"}:
            raise ValueError("budget_scope must be 'decode' or 'total'")
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if self.observation_window <= 0:
            raise ValueError("observation_window must be positive")
        if self.delay_window < self.observation_window:
            raise ValueError("delay_window must be at least observation_window")
        if self.budget <= self.delay_window:
            raise ValueError("budget must be greater than delay_window")
        if self.first_tokens < 0:
            raise ValueError("first_tokens must be non-negative")
        if self.redundancy_chunk_size <= 0:
            raise ValueError("redundancy_chunk_size must be positive")
        if self.selection_scope not in {"shared", "per_head"}:
            raise ValueError("selection_scope must be 'shared' or 'per_head'")
        if self.redundancy_neighborhood not in {"local", "global"}:
            raise ValueError("redundancy_neighborhood must be 'local' or 'global'")
        if self.redundancy_radius <= 0:
            raise ValueError("redundancy_radius must be positive")
        if self.selection_scope == "per_head" and self.redundancy_neighborhood != "global":
            raise ValueError("the recovered per_head compatibility path requires global redundancy")
        if self.budget_scope == "total" and self.first_tokens > self.budget - self.delay_window:
            raise ValueError("initial-token protection exceeds the eligible budget")


class AvoKVE:
    """Payload-aware, delayed-eligibility KV-cache eviction.

    The default averages raw features across layers and heads before ranking,
    then applies one keep set everywhere. ``per_head`` explicitly selects the
    recovered compatibility variant, not the paper's aggregation rule.
    """

    method_name = "avokv_e"

    def __init__(
        self,
        budget: int,
        budget_scope: str = "decode",
        window_size: int | None = None,
        observation_window: int = 8,
        delay_window: int | None = None,
        lambda_p: float = 0.1,
        lambda_r: float = 0.9,
        lambda_b: float = 0.05,
        first_tokens: int = 4,
        redundancy_chunk_size: int = 256,
        selection_scope: str = "shared",
        redundancy_neighborhood: str = "local",
        redundancy_radius: int = 8,
        **_: object,
    ) -> None:
        if window_size is not None:
            observation_window = int(window_size)
        if delay_window is None:
            delay_window = observation_window
        self.config = AvoKVEConfig(
            budget=int(budget),
            budget_scope=str(budget_scope),
            observation_window=int(observation_window),
            delay_window=int(delay_window),
            lambda_p=float(lambda_p),
            lambda_r=float(lambda_r),
            lambda_b=float(lambda_b),
            first_tokens=int(first_tokens),
            redundancy_chunk_size=int(redundancy_chunk_size),
            selection_scope=str(selection_scope),
            redundancy_neighborhood=str(redundancy_neighborhood),
            redundancy_radius=int(redundancy_radius),
        )
        self.prefill_tokens = 0
        self.layer_idx: int | None = None
        self.last_keep_indices: torch.Tensor | None = None

    @property
    def window_size(self) -> int:
        return self.config.observation_window

    def reset_for_prefill(self, prefill_tokens: int) -> None:
        self.prefill_tokens = int(prefill_tokens)
        self.last_keep_indices = None

    def total_budget(self) -> int:
        if self.config.budget_scope == "decode":
            return self.prefill_tokens + self.config.budget
        return self.config.budget

    def score_components(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
        candidate_count: int,
        positions: torch.Tensor | None = None,
        *,
        raw_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Compute raw, normalized, and combined scores for eligible entries."""
        if candidate_count <= 0 or candidate_count > key_states.shape[-2]:
            raise ValueError("candidate_count must index a non-empty cache prefix")

        observation = query_states[..., -self.config.observation_window :, :]
        attention = causal_attention_probabilities(observation, key_states)
        read_pressure = attention[..., :candidate_count].mean(dim=(2, 3))

        candidate_keys = key_states[..., :candidate_count, :]
        if self.config.redundancy_neighborhood == "local":
            redundancy = local_cosine_redundancy(
                candidate_keys,
                None if positions is None else positions[:candidate_count],
                radius=self.config.redundancy_radius,
            )
        else:
            redundancy = max_cosine_redundancy(
                candidate_keys, chunk_size=self.config.redundancy_chunk_size
            )

        candidate_values = value_states[..., :candidate_count, :].float()
        value_norm = torch.linalg.vector_norm(candidate_values, dim=-1)
        value_scale = value_norm.median(dim=-1, keepdim=True).values.clamp_min(1e-8)
        payload_potential = value_norm / value_scale

        raw = {
            "read_pressure": read_pressure,
            "redundancy": redundancy,
            "payload_potential": payload_potential,
        }
        if raw_only:
            return raw
        if self.config.selection_scope == "shared":
            raw = {name: value.mean(dim=1, keepdim=True) for name, value in raw.items()}
        return self.rank_components(raw)

    def rank_components(self, raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        read_pressure, redundancy, payload_potential = (
            raw[name] for name in ("read_pressure", "redundancy", "payload_potential")
        )
        read_rank = percentile_rank(read_pressure)
        redundancy_rank = percentile_rank(redundancy)
        payload_rank = percentile_rank(payload_potential)
        score = (
            self.config.lambda_p * read_rank
            - self.config.lambda_r * redundancy_rank
            + self.config.lambda_b * payload_rank
        )

        protected_initial = min(self.config.first_tokens, score.shape[-1])
        if protected_initial:
            score = score.clone()
            score[..., :protected_initial] = float("inf")

        return {
            "read_pressure": read_pressure,
            "redundancy": redundancy,
            "payload_potential": payload_potential,
            "read_rank": read_rank,
            "redundancy_rank": redundancy_rank,
            "payload_rank": payload_rank,
            "score": score,
        }

    def update_kv(
        self,
        key_states: torch.Tensor,
        query_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape != value_states.shape:
            raise ValueError("key_states and value_states must have identical shapes")
        if key_states.ndim != 4:
            raise ValueError("KV states must have shape [B, Hkv, T, D]")
        if key_states.shape[0] != 1:
            raise ValueError("AvoKV-E compressed generation currently requires batch_size=1")

        before_len = key_states.shape[-2]
        total_budget = self.total_budget()
        if before_len <= total_budget:
            self._record_event(before_len, before_len)
            return key_states, value_states

        protected_count = min(self.config.delay_window, before_len)
        candidate_count = before_len - protected_count
        candidate_slots = total_budget - protected_count
        if candidate_slots <= 0:
            raise RuntimeError("delay window leaves no candidate slots")

        components = self.score_components(
            key_states,
            query_states,
            value_states,
            candidate_count,
        )
        score = components["score"]

        if self.config.budget_scope == "decode" and self.prefill_tokens:
            prompt_count = min(self.prefill_tokens, candidate_count)
            if prompt_count > candidate_slots:
                raise RuntimeError("decode budget cannot retain the complete prompt")
            score = score.clone()
            score[..., :prompt_count] = float("inf")

        keep = score.topk(candidate_slots, dim=-1).indices.sort(dim=-1).values
        self.last_keep_indices = keep.detach()
        keep = keep.expand(key_states.shape[0], key_states.shape[1], -1)
        gather_index = keep.unsqueeze(-1).expand(-1, -1, -1, key_states.shape[-1])
        kept_keys = key_states[..., :candidate_count, :].gather(2, gather_index)
        kept_values = value_states[..., :candidate_count, :].gather(2, gather_index)

        compressed_keys = torch.cat([kept_keys, key_states[..., -protected_count:, :]], dim=-2)
        compressed_values = torch.cat(
            [kept_values, value_states[..., -protected_count:, :]], dim=-2
        )
        self._record_event(before_len, compressed_keys.shape[-2])
        return compressed_keys, compressed_values

    def _record_event(self, before_len: int, after_len: int) -> None:
        decode_len = after_len
        if self.config.budget_scope == "decode":
            decode_len = max(after_len - self.prefill_tokens, 0)
        record_kv_event(
            method=self.method_name,
            layer_idx=self.layer_idx,
            before_len=before_len,
            after_len=after_len,
            budget=self.config.budget,
            window_size=self.config.observation_window,
            observation_window=self.config.observation_window,
            delay_window=self.config.delay_window,
            budget_scope=self.config.budget_scope,
            configured_decode_budget=self.config.budget,
            total_budget=self.total_budget(),
            prefill_tokens=self.prefill_tokens,
            decode_kv_len=decode_len,
            dense_tokens=after_len,
            quantized_tokens=0,
            estimated_effective_kv_tokens=float(after_len),
            lambda_p=self.config.lambda_p,
            lambda_r=self.config.lambda_r,
            lambda_b=self.config.lambda_b,
            first_tokens=self.config.first_tokens,
            selection_scope=self.config.selection_scope,
            redundancy_neighborhood=self.config.redundancy_neighborhood,
            redundancy_radius=self.config.redundancy_radius,
        )

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self.config)


def update_shared_cache(
    policies: list[AvoKVE],
    states: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    positions: torch.Tensor,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
    """Average raw layer/head features, rank once, and gather a common keep set."""
    if not policies or len(policies) != len(states):
        raise ValueError("one policy and (key, query, value) tuple is required per layer")
    policy = policies[0]
    before = states[0][0].shape[-2]
    if any(p.config != policy.config for p in policies):
        raise ValueError("shared selection requires identical policy configurations")
    if any(k.shape != v.shape or k.shape[0] != 1 or k.shape[-2] != before for k, _, v in states):
        raise ValueError("shared selection requires aligned, unpadded batch-size-1 caches")
    if positions.shape != (before,):
        raise ValueError("absolute cache positions must align with KV states")
    total_budget = policy.total_budget()
    if before <= total_budget:
        for p in policies:
            p._record_event(before, before)
        return [(k, v) for k, _, v in states], positions

    candidate_count = before - policy.config.delay_window
    candidate_slots = total_budget - policy.config.delay_window
    device = states[0][0].device
    totals: dict[str, torch.Tensor] = {}
    for p, (keys, queries, values) in zip(policies, states, strict=True):
        raw = p.score_components(keys, queries, values, candidate_count, positions, raw_only=True)
        for name, feature in raw.items():
            averaged = feature.mean(dim=1, keepdim=True).to(device)
            totals[name] = totals.get(name, 0) + averaged / len(policies)
    score = policy.rank_components(totals)["score"]
    if policy.config.budget_scope == "decode":
        score[..., : policy.prefill_tokens] = float("inf")
    if (
        max(
            policy.config.first_tokens,
            policy.prefill_tokens if policy.config.budget_scope == "decode" else 0,
        )
        > candidate_slots
    ):
        raise ValueError("protected prefix exceeds the candidate budget")
    keep = score.topk(candidate_slots, dim=-1).indices.sort(dim=-1).values.flatten()
    tail = torch.arange(candidate_count, before, device=device)
    all_keep = torch.cat((keep, tail))
    compressed = []
    for p, (keys, _, values) in zip(policies, states, strict=True):
        p.last_keep_indices = keep.view(1, 1, -1).detach()
        indices = all_keep.to(keys.device)
        compressed.append((keys.index_select(-2, indices), values.index_select(-2, indices)))
        p._record_event(before, total_budget)
    return compressed, positions.index_select(0, all_keep.to(positions.device))


__all__ = [
    "AvoKVE",
    "AvoKVEConfig",
    "causal_attention_probabilities",
    "max_cosine_redundancy",
    "local_cosine_redundancy",
    "percentile_rank",
    "update_shared_cache",
]
