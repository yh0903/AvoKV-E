from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from avokv.hf.monkeypatch import restore_model_family


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generation_refuses_to_overwrite_its_dataset_even_with_overwrite(tmp_path) -> None:
    script = load_script("run_reasoning")
    dataset = tmp_path / "samples.jsonl"
    original = '{"question":"1+1","answer":"2"}\n'
    dataset.write_text(original)
    args = Namespace(
        batch_size=1,
        device_map="cpu",
        dataset=dataset,
        output=dataset,
        metrics=None,
        events=None,
        overwrite=True,
    )
    with pytest.raises(ValueError, match="input dataset"):
        script.run(args)
    assert dataset.read_text() == original


@pytest.mark.parametrize("collision", ["input_metrics", "input_details", "metrics_details"])
def test_evaluation_refuses_artifact_path_collisions(tmp_path, monkeypatch, collision) -> None:
    script = load_script("evaluate_math")
    source = tmp_path / "generations.jsonl"
    original = '{"output":"Final answer: 2","answer":"2"}\n'
    source.write_text(original)
    metrics, details = tmp_path / "metrics.json", tmp_path / "details.jsonl"
    if collision == "input_metrics":
        metrics = source
    elif collision == "input_details":
        details = source
    else:
        details = metrics
    monkeypatch.setattr(
        script, "parse_args", lambda: Namespace(input=source, metrics=metrics, details=details)
    )
    with pytest.raises(ValueError, match="different paths"):
        script.main()
    assert source.read_text() == original


@pytest.mark.parametrize("method,budget", [("fullkv", 64), ("avokv_e", 64), ("avokv_e", 4)])
def test_cpu_cli_roundtrip_reports_actual_cache_lengths(
    tmp_path, monkeypatch, method, budget
) -> None:
    script = load_script("run_reasoning")
    restore_model_family("llama")
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        eos_token_id=None,
        pad_token_id=0,
    )
    config._attn_implementation = "sdpa"

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = None

        def __call__(self, *args, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 7, 9]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
            }

        def encode(self, *args, **kwargs):
            return [11]

        def decode(self, *args, **kwargs):
            return "Final answer: 2"

    monkeypatch.setattr(script.AutoConfig, "from_pretrained", lambda *a, **k: config)
    monkeypatch.setattr(script.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(
        script.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: LlamaForCausalLM(config)
    )
    dataset, output = tmp_path / "samples.jsonl", tmp_path / "generations.jsonl"
    dataset.write_text('{"question":"1+1","answer":"2"}\n')
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_reasoning.py",
            "--model",
            "tiny",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--method",
            method,
            "--kv-budget",
            str(budget),
            "--device-map",
            "cpu",
            "--dtype",
            "float32",
            "--max-new-tokens",
            "10",
            "--do-sample",
            "false",
            "--observation-window",
            "2",
            "--delay-window",
            "2",
            "--first-tokens",
            "1",
            "--compression-interval",
            "2",
        ],
    )
    try:
        script.run(script.parse_args())
    finally:
        restore_model_family("llama")
    row = json.loads(output.read_text())
    assert row["output_tokens"] == 10
    assert row["kv_cache_estimate"]["full_tokens"] == 12
    assert row["run_config"]["selection_scope"] == "shared"
    if method == "avokv_e" and budget == 4:
        assert row["kv_cache_estimate"]["checkpoint_max_tokens_after"] == 7
        assert row["kv_cache_estimate"]["active_tokens_at_last_forward"] == 8
        assert row["kv_cache_estimate"]["active_tokens"] == 9
    else:
        assert row["kv_cache_estimate"]["active_tokens"] == 12
