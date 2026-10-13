#!/usr/bin/env python3
"""Generate MATH/AIME solutions with FullKV or AvoKV-E.

The Hugging Face integration is adapted from the Apache-2.0 R-KV runner and
has been modified for AvoKV-E. See ``NOTICE`` for attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from avokv.compression import (
    enable_metrics,
    get_kv_events,
    reset_metrics,
    summarize_metrics,
)

GENERIC_PROMPT = (
    "You are given a math problem.\n\n"
    "Problem: {question}\n\n"
    "Solve the problem step by step, then provide the final answer in the "
    "format: Final answer: \\boxed{{}}"
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_seed(base_seed: int, source_index: Any, candidate_index: int) -> int:
    payload = json.dumps(
        [int(base_seed), str(source_index), int(candidate_index)],
        separators=(",", ":"),
    ).encode("utf-8")
    return zlib.crc32(payload) & 0x7FFFFFFF


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def question_from_row(row: dict[str, Any]) -> str:
    for key in ("problem", "question"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError("each input row must contain a non-empty 'problem' or 'question'")


def build_prompt(tokenizer, question: str, style: str) -> str:
    if style == "generic":
        return GENERIC_PROMPT.format(question=question)
    if style == "qwen3-chat":
        messages = [
            {
                "role": "system",
                "content": (
                    "Solve the math problem step by step. End with the final "
                    "answer in the format: Final answer: \\boxed{}"
                ),
            },
            {"role": "user", "content": question},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError(f"unsupported prompt style: {style}")


def resolve_prompt_style(requested: str, model_type: str) -> str:
    if requested != "auto":
        return requested
    return "qwen3-chat" if model_type == "qwen3" else "generic"


def resolve_delay_window(budget: int, requested: int | None) -> int:
    if requested is not None:
        return requested
    return 32 if budget <= 512 else 8


def compression_config(args: argparse.Namespace, delay_window: int) -> dict[str, Any]:
    return {
        "method": "avokv_e",
        "method_config": {
            "budget": args.kv_budget,
            "budget_scope": args.budget_scope,
            "window_size": args.observation_window,
            "observation_window": args.observation_window,
            "delay_window": delay_window,
            "lambda_p": args.lambda_p,
            "lambda_r": args.lambda_r,
            "lambda_b": args.lambda_b,
            "first_tokens": args.first_tokens,
            "redundancy_chunk_size": args.redundancy_chunk_size,
            "selection_scope": args.selection_scope,
            "redundancy_neighborhood": args.redundancy_neighborhood,
            "redundancy_radius": args.redundancy_radius,
        },
        "compression": None,
        "update_kv": True,
    }


def encode_last_token_ids(tokenizer, texts: list[str]) -> list[int]:
    token_ids = []
    for text in texts:
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded:
            token_ids.append(encoded[-1])
    return sorted(set(token_ids))


def estimate_kv_bytes(model, token_count: int) -> int:
    config = model.config
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    return int(
        token_count
        * config.num_hidden_layers
        * 2
        * kv_heads
        * head_dim
        * next(model.parameters()).element_size()
    )


def write_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> None:
    if args.batch_size != 1:
        raise ValueError("the release runner requires --batch-size 1")
    if not torch.cuda.is_available() and args.device_map != "cpu":
        raise RuntimeError("CUDA is unavailable; pass --device-map cpu only for tiny smoke tests")

    output_path = args.output
    metrics_path = args.metrics or output_path.with_suffix(".metrics.jsonl")
    events_path = args.events
    artifact_paths = [output_path, metrics_path]
    if events_path is not None:
        artifact_paths.append(events_path)
    resolved_paths = [path.resolve() for path in artifact_paths]
    if args.dataset.resolve() in resolved_paths:
        raise ValueError("generation artifacts must not overwrite the input dataset")
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("--output, --metrics, and --events must refer to different files")
    existing = [path for path in artifact_paths if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite {names}; pass --overwrite")
    rows = load_jsonl(args.dataset)
    dataset_hash = sha256_file(args.dataset)

    model_config = AutoConfig.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model_type = str(model_config.model_type).lower()
    delay_window = resolve_delay_window(args.kv_budget, args.delay_window)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        use_fast=True,
        padding_side="left",
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.method == "avokv_e":
        from avokv.hf import patch_model_family

        patch_model_family(
            model_type,
            compression_config(args, delay_window),
        )

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
        use_cache=True,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    model.config.update(
        {
            "divide_method": "step_length",
            "divide_length": args.compression_interval,
            "compression_content": "all",
            "batch_invariant_compression": False,
        }
    )
    if args.method == "avokv_e":
        model.newline_token_ids = encode_last_token_ids(tokenizer, ["\n"])
        model.after_think_token_ids = encode_last_token_ids(tokenizer, ["</think>"])

    start = args.start_index
    stop = len(rows) if args.limit is None else min(len(rows), start + args.limit)
    indexed_rows = list(enumerate(rows[start:stop], start=start))
    prompt_style = resolve_prompt_style(args.prompt_style, model_type)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if events_path is not None:
        events_path.parent.mkdir(parents=True, exist_ok=True)

    enable_metrics(args.method == "avokv_e")
    run_config = {
        "model": args.model,
        "model_revision": args.revision,
        "model_type": model_type,
        "dataset": str(args.dataset),
        "dataset_sha256": dataset_hash,
        "method": args.method,
        "kv_budget": args.kv_budget,
        "budget_scope": args.budget_scope,
        "observation_window": args.observation_window,
        "delay_window": delay_window,
        "lambda_p": args.lambda_p,
        "lambda_r": args.lambda_r,
        "lambda_b": args.lambda_b,
        "first_tokens": args.first_tokens,
        "compression_interval": args.compression_interval,
        "redundancy_chunk_size": args.redundancy_chunk_size,
        "selection_scope": args.selection_scope,
        "redundancy_neighborhood": args.redundancy_neighborhood,
        "redundancy_radius": args.redundancy_radius,
        "max_total_tokens": args.max_total_tokens,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "samples_per_problem": args.samples_per_problem,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "prompt_style": prompt_style,
        "attn_implementation": args.attn_implementation,
        "dtype": args.dtype,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }

    with (
        output_path.open("w", encoding="utf-8") as output_handle,
        metrics_path.open("w", encoding="utf-8") as metrics_handle,
    ):
        events_handle = events_path.open("w", encoding="utf-8") if events_path is not None else None
        try:
            for source_index, row in tqdm(indexed_rows, desc="Generate"):
                question = question_from_row(row)
                prompt = build_prompt(tokenizer, question, prompt_style)
                tokenized = tokenizer(
                    prompt, return_tensors="pt", add_special_tokens=prompt_style == "generic"
                )
                model_device = next(model.parameters()).device
                tokenized = {name: value.to(model_device) for name, value in tokenized.items()}
                prompt_tokens = int(tokenized["input_ids"].shape[-1])
                max_new_tokens = args.max_total_tokens - prompt_tokens
                if args.max_new_tokens is not None:
                    max_new_tokens = min(max_new_tokens, args.max_new_tokens)
                if max_new_tokens <= 0:
                    raise ValueError(
                        f"prompt at source index {source_index} has {prompt_tokens} tokens, "
                        f"which reaches --max-total-tokens={args.max_total_tokens}"
                    )

                for candidate_index in range(args.samples_per_problem):
                    generation_seed = sample_seed(
                        args.seed,
                        row.get("source_index", source_index),
                        candidate_index,
                    )
                    set_seed(generation_seed)
                    reset_metrics()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()

                    generation_kwargs: dict[str, Any] = {
                        "max_new_tokens": max_new_tokens,
                        "do_sample": args.do_sample,
                        "num_beams": 1,
                        "use_cache": True,
                        "return_dict_in_generate": True,
                        "pad_token_id": tokenizer.pad_token_id,
                    }
                    if tokenizer.eos_token_id is not None:
                        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
                    if args.do_sample:
                        generation_kwargs.update(
                            {"temperature": args.temperature, "top_p": args.top_p}
                        )

                    started = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(**tokenized, **generation_kwargs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started

                    generated_ids = generated.sequences[0, prompt_tokens:]
                    output_tokens = int(generated_ids.numel())
                    total_tokens = prompt_tokens + output_tokens
                    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    kv_summary = summarize_metrics()
                    final_cache_tokens = int(generated.past_key_values.get_seq_length())
                    full_cache_tokens = prompt_tokens + max(output_tokens - 1, 0)
                    effective_tokens = max(
                        prompt_tokens,
                        final_cache_tokens,
                        int(kv_summary.get("max_kv_len_before", 0)),
                    )
                    peak_allocated = (
                        int(torch.cuda.max_memory_allocated())
                        if torch.cuda.is_available()
                        else None
                    )

                    result = dict(row)
                    result.update(
                        {
                            "idx": row.get("idx", row.get("source_index", source_index)),
                            "source_index": row.get("source_index", source_index),
                            "candidate_index": candidate_index,
                            "generation_seed": generation_seed,
                            "prompt": prompt,
                            "output": text,
                            "prompt_tokens": prompt_tokens,
                            "output_tokens": output_tokens,
                            "total_tokens": total_tokens,
                            "hit_max_new_tokens": output_tokens >= max_new_tokens,
                            "elapsed_seconds": elapsed,
                            "tokens_per_second": output_tokens / elapsed if elapsed else None,
                            "peak_cuda_allocated_bytes": peak_allocated,
                            "kv_cache_estimate": {
                                "full_tokens": full_cache_tokens,
                                "active_tokens": effective_tokens,
                                "active_tokens_at_last_forward": final_cache_tokens,
                                "checkpoint_max_tokens_after": kv_summary.get("max_kv_len_after", 0),
                                "full_bytes": estimate_kv_bytes(model, full_cache_tokens),
                                "active_bytes": estimate_kv_bytes(model, effective_tokens),
                            },
                            "kv_metrics": kv_summary,
                            "run_config": run_config,
                        }
                    )
                    write_jsonl(output_handle, result)
                    write_jsonl(
                        metrics_handle,
                        {
                            "source_index": result["source_index"],
                            "candidate_index": candidate_index,
                            "generation_seed": generation_seed,
                            "prompt_tokens": prompt_tokens,
                            "output_tokens": output_tokens,
                            "elapsed_seconds": elapsed,
                            "tokens_per_second": result["tokens_per_second"],
                            "peak_cuda_allocated_bytes": peak_allocated,
                            "kv_cache_estimate": result["kv_cache_estimate"],
                            "kv_metrics": kv_summary,
                            "run_config": run_config,
                        },
                    )
                    if events_handle is not None:
                        write_jsonl(
                            events_handle,
                            {
                                "source_index": result["source_index"],
                                "candidate_index": candidate_index,
                                "kv_events": get_kv_events(),
                            },
                        )
                    del generated
        finally:
            if events_handle is not None:
                events_handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--revision",
        help="Optional immutable model revision or commit hash.",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--method", choices=["fullkv", "avokv_e"], default="avokv_e")
    parser.add_argument("--kv-budget", type=int, default=512)
    parser.add_argument(
        "--budget-scope",
        choices=["decode", "total"],
        default="decode",
        help="Paper release default: retain the prompt and budget decode-time KV separately.",
    )
    parser.add_argument("--observation-window", type=int, default=8)
    parser.add_argument(
        "--delay-window",
        type=int,
        help="Default is 32 for budgets <=512 and 8 otherwise.",
    )
    parser.add_argument("--lambda-p", type=float, default=0.1)
    parser.add_argument("--lambda-r", type=float, default=0.9)
    parser.add_argument("--lambda-b", type=float, default=0.05)
    parser.add_argument("--first-tokens", type=int, default=4)
    parser.add_argument("--compression-interval", type=int, default=128)
    parser.add_argument("--redundancy-chunk-size", type=int, default=256)
    parser.add_argument("--selection-scope", choices=["shared", "per_head"], default="shared")
    parser.add_argument("--redundancy-neighborhood", choices=["local", "global"], default="local")
    parser.add_argument("--redundancy-radius", type=int, default=8)
    parser.add_argument("--max-total-tokens", type=int, default=32768)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Optional decode cap applied in addition to --max-total-tokens.",
    )
    parser.add_argument("--samples-per-problem", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do-sample", type=str2bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--prompt-style",
        choices=["auto", "generic", "qwen3-chat"],
        default="auto",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Reserved for provenance; this release runner requires 1.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
