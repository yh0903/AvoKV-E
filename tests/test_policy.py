from __future__ import annotations

import pytest
import torch

from avokv.compression.policy import (
    AvoKVE,
    causal_attention_probabilities,
    local_cosine_redundancy,
    max_cosine_redundancy,
    percentile_rank,
    update_shared_cache,
)


def test_percentile_rank_averages_exact_ties() -> None:
    values = torch.tensor([[[1.0, 2.0, 2.0, 4.0]]])
    ranks = percentile_rank(values)
    expected = torch.tensor([[[0.0, 0.5, 0.5, 1.0]]])
    torch.testing.assert_close(ranks, expected)


def test_causal_attention_masks_keys_created_after_each_query() -> None:
    queries = torch.tensor([[[[1.0], [1.0]]]])
    keys = torch.tensor([[[[0.0], [0.0], [0.0], [20.0]]]])

    probabilities = causal_attention_probabilities(queries, keys)

    assert probabilities.shape == (1, 1, 1, 2, 4)
    assert probabilities[0, 0, 0, 0, 3].item() == 0.0
    assert probabilities[0, 0, 0, 1, 3].item() > 0.999


def test_grouped_query_attention_is_preserved_before_head_average() -> None:
    queries = torch.tensor([[[[1.0]], [[-1.0]]]])
    keys = torch.tensor([[[[1.0], [-1.0]]]])

    probabilities = causal_attention_probabilities(queries, keys)

    assert probabilities.shape == (1, 1, 2, 1, 2)
    assert probabilities[0, 0, 0, 0, 0] > 0.8
    assert probabilities[0, 0, 1, 0, 1] > 0.8


def test_chunked_redundancy_matches_dense_cosine_max() -> None:
    generator = torch.Generator().manual_seed(7)
    keys = torch.randn(2, 3, 11, 5, generator=generator)
    normalized = torch.nn.functional.normalize(keys.float(), dim=-1)
    dense = torch.matmul(normalized, normalized.transpose(-1, -2))
    diagonal = torch.eye(11, dtype=torch.bool).view(1, 1, 11, 11)
    expected = dense.masked_fill(diagonal, float("-inf")).amax(dim=-1)

    actual = max_cosine_redundancy(keys, chunk_size=4)

    torch.testing.assert_close(actual, expected)


def test_payload_term_is_independent_of_read_pressure() -> None:
    policy = AvoKVE(
        budget=6,
        observation_window=2,
        delay_window=2,
        lambda_p=0.0,
        lambda_r=0.0,
        lambda_b=1.0,
        first_tokens=0,
    )
    keys = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]]])
    queries_a = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    queries_b = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]])
    values = torch.tensor([[[[1.0, 0.0], [4.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]])

    score_a = policy.score_components(keys, queries_a, values, 2)["score"]
    score_b = policy.score_components(keys, queries_b, values, 2)["score"]

    torch.testing.assert_close(score_a, score_b)
    assert score_a[0, 0, 1] > score_a[0, 0, 0]


def test_decode_budget_keeps_prompt_outside_decode_allowance() -> None:
    policy = AvoKVE(
        budget=4,
        budget_scope="decode",
        observation_window=2,
        delay_window=2,
        first_tokens=0,
    )
    policy.reset_for_prefill(3)
    keys = torch.randn(1, 1, 9, 4, generator=torch.Generator().manual_seed(3))
    values = torch.randn(1, 1, 9, 4, generator=torch.Generator().manual_seed(4))
    queries = torch.randn(1, 1, 2, 4, generator=torch.Generator().manual_seed(5))

    compressed_keys, compressed_values = policy.update_kv(keys, queries, values)

    assert compressed_keys.shape[-2] == 7
    assert compressed_values.shape[-2] == 7
    assert set(range(3)).issubset(set(policy.last_keep_indices.flatten().tolist()))
    torch.testing.assert_close(compressed_keys[..., -2:, :], keys[..., -2:, :])


def test_total_budget_is_enforced_exactly() -> None:
    policy = AvoKVE(
        budget=5,
        budget_scope="total",
        observation_window=2,
        delay_window=2,
        first_tokens=1,
    )
    keys = torch.randn(1, 2, 8, 4, generator=torch.Generator().manual_seed(8))
    values = torch.randn(1, 2, 8, 4, generator=torch.Generator().manual_seed(9))
    queries = torch.randn(1, 4, 2, 4, generator=torch.Generator().manual_seed(10))

    compressed_keys, _ = policy.update_kv(keys, queries, values)

    assert compressed_keys.shape == (1, 2, 5, 4)
    assert torch.all(policy.last_keep_indices[..., 0] == 0)


def test_policy_rejects_noncausal_window_configuration() -> None:
    with pytest.raises(ValueError, match="at least observation_window"):
        AvoKVE(budget=128, observation_window=16, delay_window=8)


def test_compressed_generation_requires_batch_size_one() -> None:
    policy = AvoKVE(budget=4, observation_window=2, delay_window=2)
    keys = torch.randn(2, 1, 8, 4)
    values = torch.randn(2, 1, 8, 4)
    queries = torch.randn(2, 1, 2, 4)

    with pytest.raises(ValueError, match="batch_size=1"):
        policy.update_kv(keys, queries, values)


def test_local_redundancy_uses_absolute_positions_after_eviction() -> None:
    keys = torch.tensor([[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]])
    positions = torch.tensor([0, 10, 11, 12])
    actual = local_cosine_redundancy(keys, positions, radius=1)
    torch.testing.assert_close(actual, torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]))


def test_shared_selection_averages_raw_features_before_ranking(monkeypatch) -> None:
    policies = [
        AvoKVE(
            budget=4,
            observation_window=2,
            delay_window=2,
            first_tokens=0,
            lambda_p=1,
            lambda_r=0,
            lambda_b=0,
        )
        for _ in range(2)
    ]
    features = [
        torch.tensor([[[100.0, 2.0, 1.0, 0.0], [0.0, 4.0, 3.0, 2.0]]]),
        torch.tensor([[[0.0, 4.0, 3.0, 2.0], [0.0, 4.0, 3.0, 2.0]]]),
    ]
    for policy, feature in zip(policies, features, strict=True):

        def raw(*args, feature=feature, **kwargs):
            return {
                "read_pressure": feature,
                "redundancy": feature * 0,
                "payload_potential": feature * 0,
            }

        monkeypatch.setattr(policy, "score_components", raw)
    keys = torch.arange(12.0).reshape(1, 2, 6, 1)
    states = [(keys, keys[..., -2:, :], keys + i) for i in range(2)]
    compressed, positions = update_shared_cache(policies, states, torch.arange(6))
    # Ranking each head first would choose candidates 1 and 2 instead of 0 and 1.
    assert positions.tolist() == [0, 1, 4, 5]
    for policy, (kept_keys, _) in zip(policies, compressed, strict=True):
        assert policy.last_keep_indices.flatten().tolist() == [0, 1]
        torch.testing.assert_close(kept_keys, keys.index_select(-2, positions))


def test_shared_selection_retains_prompt_and_delay_across_layers() -> None:
    policies = [AvoKVE(budget=4, observation_window=2, delay_window=2) for _ in range(3)]
    generator = torch.Generator().manual_seed(31)
    for policy in policies:
        policy.reset_for_prefill(5)
    states = [
        (
            torch.randn(1, 2, 12, 4, generator=generator),
            torch.randn(1, 4, 2, 4, generator=generator),
            torch.randn(1, 2, 12, 4, generator=generator),
        )
        for _ in policies
    ]
    compressed, positions = update_shared_cache(policies, states, torch.arange(12))
    assert positions.shape == (9,)
    assert positions[:5].tolist() == list(range(5))
    assert positions[-2:].tolist() == [10, 11]
    for keys, values in compressed:
        assert keys.shape == values.shape == (1, 2, 9, 4)
