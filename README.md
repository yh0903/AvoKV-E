# AvoKV-E

Implementation of **AvoKV-E: Payload-Aware KV Cache
Eviction for Long Reasoning**.

[[Paper](https://arxiv.org/abs/2610.03007)]
[[Citation](#citation)]

AvoKV-E is a training-free eviction policy for decode-time KV caches. It
separates two decisions that routing-only policies conflate:

1. **When may a state be judged?** Recent states remain in a protected delay
   buffer until later queries have had a causal opportunity to read them.
2. **Which eligible states should remain?** Older entries are ranked by
   candidate-relative read pressure, key redundancy, and value-payload
   potential.

For an eligible entry `i`, the keep score is

```text
S_i = lambda_p * rank(read_i)
    - lambda_r * rank(redundancy_i)
    + lambda_b * rank(value_norm_i)
```

The default weights are `(0.1, 0.9, 0.05)`. The payload term is independent of
read pressure so a low-attention state can still be rescued when its value
payload is unusually salient.

## Installation

Python 3.10+ and a recent CUDA-capable PyTorch environment are recommended.
The Hugging Face integration uses `transformers==5.8.0`.

```bash
git clone https://github.com/yh0903/AvoKV-E.git
cd AvoKV-E
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[eval,data]"
```

FlashAttention is optional. The default runner uses PyTorch SDPA because it is
the most portable setup.

## Prepare MATH-500

Download the public `HuggingFaceH4/MATH-500` split:

```bash
python scripts/prepare_math500.py --output data/math500.jsonl
```

For AIME, provide a JSONL file with one object per problem and the fields
`question` and `answer`.

## Run AvoKV-E

DeepSeek-R1-Distill-Llama-8B on MATH-500 at decode budget 512:

```bash
python scripts/run_reasoning.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --dataset data/math500.jsonl \
  --output outputs/deepseek_math500_avokv_e_b512.jsonl \
  --method avokv_e \
  --kv-budget 512 \
  --budget-scope decode \
  --max-total-tokens 32768
```

The paper delay rule is automatic: `32` at budgets up to 512 and `8` at
budgets of 768 or larger. Override it explicitly with `--delay-window`.

Run the matched FullKV reference through the same generation loop:

```bash
python scripts/run_reasoning.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --dataset data/math500.jsonl \
  --output outputs/deepseek_math500_fullkv.jsonl \
  --method fullkv \
  --max-total-tokens 32768
```

Evaluate either output:

```bash
python scripts/evaluate_math.py \
  --input outputs/deepseek_math500_avokv_e_b512.jsonl
```

For Qwen3-14B, change `--model` to `Qwen/Qwen3-14B`; the runner selects its
chat template and Qwen3 attention patch automatically. For the paper's
AIME-25 protocol, add `--samples-per-problem 16`.

## Settings

Default settings are recorded in [`configs/paper.json`](configs/paper.json):

| Setting | Value |
| --- | ---: |
| Observation window | 8 |
| Delay window | 32 if `B <= 512`, else 8 |
| Compression interval | 128 generated tokens |
| Score weights | `(0.1, 0.9, 0.05)` |
| Protected initial entries | 4 |
| Feature aggregation | Raw features averaged across layers and heads, then ranked |
| Keep set | Shared across layers and heads |
| Redundancy neighborhood | Eligible entries within absolute token radius 8 |
| Maximum total length | 32K tokens |
| Sampling | temperature `0.6`, top-p `0.95` |
| Evaluation batch size | 1 |

`--budget-scope decode` retains the prompt outside the stated decode budget.
Prompt entries participate in feature statistics but are forced into the keep
set rather than evicted.

The configured budget is restored at each periodic compression checkpoint.
Between checkpoints, the cache can temporarily grow by at most
`compression_interval - 1` generated entries.

The policy computes strictly causal read pressure: an earlier query in the
observation window cannot attend to a key created later. At each checkpoint,
raw read, redundancy, and median-scaled value features are averaged across
layers and heads **before** percentile normalization. One chronological keep
set is applied to every layer/head after the checkpoint forward completes.
Local redundancy excludes the candidate itself and uses surviving eligible
entries within absolute token distance 8. A token with no surviving neighbor
has redundancy zero; eviction gaps do not make distant entries adjacent.

To pin a model snapshot, pass its commit through `--revision`. The
runner records that revision together with a SHA-256 digest of the input JSONL,
resolved policy settings, package versions, token counts, and cache events.

## Tests

```bash
python -m pip install -e ".[dev]"
pytest -q
ruff check src scripts tests
```

The CPU suite covers tie-aware percentile ranks, grouped-query attention,
causal masking, chunked redundancy, independent payload rescue, delayed
eligibility, decode-only budget accounting, shared feature aggregation, and
FullKV logit parity with no eviction.

## Attribution and license

The Hugging Face integration and mathematical equivalence checker are adapted
from the Apache-2.0 [R-KV](https://github.com/Zefan-Cai/R-KV) release. See
[`NOTICE`](NOTICE) for retained attribution.

The code is released under the [Apache License 2.0](LICENSE). Model and dataset
licenses remain governed by their respective providers.

## Citation

```bibtex
@article{yu2026avokve,
  title   = {AvoKV-E: Payload-Aware KV Cache Eviction for Long Reasoning},
  author  = {Yu, Han and Zhu, Wenhui and Chen, Xiwen and Wang, Zhipeng and
             Sang, Hejian and Shi, Han and Zhou, Menglin and Dong, Xuanzhao and
             Huang, Minzhou and Cai, Rui and Wang, Hao and Geramifard, Alborz},
  journal = {arXiv preprint arXiv:2610.03007},
  year    = {2026}
}
```
