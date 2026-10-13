#!/usr/bin/env python3
"""Validate real-model logits and eviction invariants, not benchmark accuracy."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from avokv.compression import AvoKVE, enable_metrics, reset_metrics, summarize_metrics
from avokv.hf import patch_model_family
from avokv.hf.monkeypatch import restore_model_family


def validate(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("this real-model validation requires CUDA")
    torch.manual_seed(13)
    family = AutoConfig.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only
    ).model_type
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=args.local_files_only
    )
    prompt = (
        tokenizer(
            "Work through the reasoning carefully. A cache entry contains a key and a value. " * 12,
            return_tensors="pt",
        )
        .input_ids[:, :64]
        .cuda()
    )
    followup = torch.randint(10, tokenizer.vocab_size, (1, args.decode_steps), device="cuda")

    def load_model():
        return AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
            local_files_only=args.local_files_only,
            revision=args.revision,
        ).eval()

    restore_model_family(family)
    reference = load_model()
    expected = []
    with torch.inference_mode():
        result = reference(input_ids=prompt, use_cache=True, logits_to_keep=1)
        expected.append(result.logits.cpu())
        for step in range(8):
            result = reference(
                input_ids=followup[:, step : step + 1],
                past_key_values=result.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            expected.append(result.logits.cpu())
    del reference, result
    gc.collect()
    torch.cuda.empty_cache()

    settings = {
        "method": "avokv_e",
        "compression": None,
        "update_kv": True,
        "method_config": {"budget": 512, "budget_scope": "decode", "delay_window": 32},
    }
    patch_model_family(family, settings)
    model = load_model()
    model.config.update(
        {
            "divide_method": "step_length",
            "divide_length": 128,
            "compression_content": "all",
            "batch_invariant_compression": False,
        }
    )
    max_error = 0.0
    with torch.inference_mode():
        result = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
        for step, target in enumerate(expected):
            max_error = max(
                max_error, (result.logits.cpu().float() - target.float()).abs().max().item()
            )
            torch.testing.assert_close(result.logits.cpu(), target, rtol=0, atol=0)
            if step < 8:
                result = model(
                    input_ids=followup[:, step : step + 1],
                    past_key_values=result.past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
    checks = []
    enable_metrics(True)
    for budget in (256, 512, 1024):
        if args.decode_steps <= budget + 128:
            raise ValueError("decode_steps must force multiple evictions at every tested budget")
        for layer_index, layer in enumerate(model.model.layers):
            layer.self_attn.kv_cluster = AvoKVE(
                budget=budget, delay_window=32 if budget <= 512 else 8
            )
            layer.self_attn.kv_cluster.layer_idx = layer_index
        reset_metrics()
        started = time.monotonic()
        with torch.inference_mode():
            result = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
            for step in range(args.decode_steps):
                result = model(
                    input_ids=followup[:, step : step + 1],
                    past_key_values=result.past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                )
                if (step + 1) % 128:
                    continue
                cache = result.past_key_values
                positions = cache.avokv_positions
                assert positions[: prompt.shape[1]].tolist() == list(range(prompt.shape[1]))
                assert positions[-1].item() == prompt.shape[1] + step
                assert bool((positions.diff() > 0).all())
                assert len(positions) <= prompt.shape[1] + budget
                delay = 32 if budget <= 512 else 8
                assert positions[-delay:].tolist() == list(
                    range(prompt.shape[1] + step + 1 - delay, prompt.shape[1] + step + 1)
                )
                keeps = [
                    layer.self_attn.kv_cluster.last_keep_indices for layer in model.model.layers
                ]
                assert all(layer.keys.shape[-2] == len(positions) for layer in cache.layers)
                if keeps[0] is not None:
                    assert all(torch.equal(keeps[0], keep) for keep in keeps[1:])
                assert bool(torch.isfinite(result.logits).all())
        summary = summarize_metrics()
        assert summary["compression_events"] > 0
        checks.append(
            {
                "budget": budget,
                "decode_steps": args.decode_steps,
                "elapsed_seconds": time.monotonic() - started,
                "passed": True,
                "kv_metrics": summary,
            }
        )
    return {
        "model": args.model_id or args.model,
        "model_revision": args.revision,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "dtype": "bfloat16",
        "attention": "sdpa",
        "batch_size": 1,
        "prompt_tokens": prompt.shape[1],
        "no_eviction_logit_max_abs_error": max_error,
        "selection_scope": "shared",
        "redundancy_neighborhood": "local",
        "redundancy_radius": 8,
        "checks": checks,
        "passed": True,
        "scope": "teacher-forced correctness validation; not benchmark accuracy",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--decode-steps", type=int, default=1280)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite validation report: {args.output}")
    report = validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
