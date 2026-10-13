from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

_enabled = False
_events = []


def enable_metrics(enabled=True):
    global _enabled
    _enabled = enabled


def reset_metrics():
    _events.clear()


def record_kv_event(
    method,
    layer_idx,
    before_len,
    after_len,
    budget,
    window_size,
    **extra,
):
    if not _enabled:
        return

    dropped = max(int(before_len) - int(after_len), 0)
    event = {
        "method": method,
        "layer_idx": None if layer_idx is None else int(layer_idx),
        "before_len": int(before_len),
        "after_len": int(after_len),
        "dropped_tokens": dropped,
        "budget": None if budget is None else int(budget),
        "window_size": None if window_size is None else int(window_size),
        "compressed": dropped > 0,
    }
    event.update(extra)
    _events.append(event)


def get_kv_events():
    return deepcopy(_events)


def summarize_metrics():
    summary = {
        "events": len(_events),
        "compression_events": 0,
        "skipped_events": 0,
        "total_dropped_tokens_across_layers": 0,
        "max_kv_len_before": 0,
        "max_kv_len_after": 0,
        "max_decode_kv_len": 0,
        "max_prefill_tokens": 0,
        "max_total_budget": 0,
        "max_estimated_effective_kv_tokens": 0.0,
        "max_quantized_tokens": 0,
        "max_dense_tokens": 0,
        "per_layer": {},
    }

    per_layer = defaultdict(
        lambda: {
            "events": 0,
            "compression_events": 0,
            "skipped_events": 0,
            "total_dropped_tokens": 0,
            "max_kv_len_before": 0,
            "max_kv_len_after": 0,
            "max_decode_kv_len": 0,
            "max_prefill_tokens": 0,
            "max_total_budget": 0,
            "max_estimated_effective_kv_tokens": 0.0,
            "max_quantized_tokens": 0,
            "max_dense_tokens": 0,
        }
    )

    for event in _events:
        if event["compressed"]:
            summary["compression_events"] += 1
        else:
            summary["skipped_events"] += 1
        summary["total_dropped_tokens_across_layers"] += event["dropped_tokens"]
        summary["max_kv_len_before"] = max(summary["max_kv_len_before"], event["before_len"])
        summary["max_kv_len_after"] = max(summary["max_kv_len_after"], event["after_len"])
        summary["max_decode_kv_len"] = max(
            summary["max_decode_kv_len"], int(event.get("decode_kv_len", event["after_len"]))
        )
        summary["max_prefill_tokens"] = max(
            summary["max_prefill_tokens"], int(event.get("prefill_tokens", 0))
        )
        summary["max_total_budget"] = max(
            summary["max_total_budget"], int(event.get("total_budget", event.get("budget") or 0))
        )
        summary["max_estimated_effective_kv_tokens"] = max(
            summary["max_estimated_effective_kv_tokens"],
            float(event.get("estimated_effective_kv_tokens", event["after_len"])),
        )
        summary["max_quantized_tokens"] = max(
            summary["max_quantized_tokens"],
            int(event.get("quantized_tokens", 0) or 0),
        )
        summary["max_dense_tokens"] = max(
            summary["max_dense_tokens"],
            int(event.get("dense_tokens", event["after_len"]) or 0),
        )

        layer_key = "unknown" if event["layer_idx"] is None else str(event["layer_idx"])
        layer_stats = per_layer[layer_key]
        layer_stats["events"] += 1
        if event["compressed"]:
            layer_stats["compression_events"] += 1
        else:
            layer_stats["skipped_events"] += 1
        layer_stats["total_dropped_tokens"] += event["dropped_tokens"]
        layer_stats["max_kv_len_before"] = max(
            layer_stats["max_kv_len_before"], event["before_len"]
        )
        layer_stats["max_kv_len_after"] = max(layer_stats["max_kv_len_after"], event["after_len"])
        layer_stats["max_decode_kv_len"] = max(
            layer_stats["max_decode_kv_len"], int(event.get("decode_kv_len", event["after_len"]))
        )
        layer_stats["max_prefill_tokens"] = max(
            layer_stats["max_prefill_tokens"], int(event.get("prefill_tokens", 0))
        )
        layer_stats["max_total_budget"] = max(
            layer_stats["max_total_budget"],
            int(event.get("total_budget", event.get("budget") or 0)),
        )
        layer_stats["max_estimated_effective_kv_tokens"] = max(
            layer_stats["max_estimated_effective_kv_tokens"],
            float(event.get("estimated_effective_kv_tokens", event["after_len"])),
        )
        layer_stats["max_quantized_tokens"] = max(
            layer_stats["max_quantized_tokens"],
            int(event.get("quantized_tokens", 0) or 0),
        )
        layer_stats["max_dense_tokens"] = max(
            layer_stats["max_dense_tokens"],
            int(event.get("dense_tokens", event["after_len"]) or 0),
        )

    summary["per_layer"] = dict(per_layer)
    return summary
