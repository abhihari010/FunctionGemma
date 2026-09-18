# FunctionGemma Intent Parser

Fine-tunes Google's `functiongemma-270m-it` to parse natural-language "Leader's Intent"
commands (Raytheon AVC "Mission Bird Dog" task) into a structured function call:
`target_color`, `constraints`, `target_location`, `action`. See `PROGRESS.md` for the
full experiment log, decisions, and results.

## Setup

Requires Python 3.11+, an NVIDIA GPU (CUDA), and a Hugging Face account with access to
`google/functiongemma-270m-it` (gated model — request access on the model card first).

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/macOS

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Log in to Hugging Face (do this from Python, not the `huggingface-cli`/`hf` CLI — its
banner text crashes with `UnicodeEncodeError` on Windows consoles):

```python
from huggingface_hub import login
login(token="<your_hf_token>")
```

Or set the `HF_TOKEN` environment variable instead.

Run all scripts from the repo root (they use relative paths like `data/eval_set.jsonl`):

```bash
cd FunctionGemma
```

## Usage

### 1. Zero-shot baseline

```bash
python scripts/00_zeroshot_baseline.py
```

### 2. Fine-tune (LoRA)

```bash
python scripts/02_finetune_lora.py
```

Saves the adapter to `checkpoints/functiongemma-270m-lora-intent/`.

### 3. Evaluate

```bash
# zero-shot base model
python scripts/03_evaluate.py --label base-zero-shot

# fine-tuned adapter
python scripts/03_evaluate.py --adapter checkpoints/functiongemma-270m-lora-intent --label finetuned-lora
```

| flag | default | description |
|---|---|---|
| `--adapter` | none | path to a LoRA adapter dir; omit for base model |
| `--label` | none | name for this run — results saved to `results/<label>.json` |
| `--max-new-tokens` | 128 | generation length cap |
| `--batch-size` | 16 | length-sorted batching; use `--batch-size 1` for true single-request latency |

### 4. Quantize to GGUF and benchmark with llama.cpp (deployment path)

Requires a [llama.cpp](https://github.com/ggml-org/llama.cpp) build (CPU or Vulkan) on your `PATH`.

```bash
# merge LoRA into the base weights first (see scripts/03_evaluate.py for the merge_and_unload() call,
# or add a save_pretrained() call to dump checkpoints/merged-fp16)

python llama.cpp/convert_hf_to_gguf.py checkpoints/merged-fp16 --outfile gguf/model-f16.gguf --outtype f16
llama-quantize.exe gguf/model-f16.gguf gguf/model-q8_0.gguf Q8_0

llama-server.exe -m gguf/model-q8_0.gguf --port 8081 -c 4096 -np 1 -ngl 99

python scripts/04_benchmark_gguf.py --port 8081 --label gguf-vulkan-q8_0
```

| flag | default | description |
|---|---|---|
| `--port` | 8080 | port `llama-server` is listening on |
| `--label` | `gguf` | name for this run — results saved to `results/<label>.json` |
| `--max-new-tokens` | 128 | generation length cap |
| `--cache-prompt` | off | reuse the shared prompt prefix across requests |

### Rebuilding the datasets

`data/eval_set.jsonl` and `data/train_set.jsonl` are already generated and locked (see
`PROGRESS.md` for why eval should not be regenerated). To rebuild from scratch:

```bash
python scripts/build_eval_set.py
python scripts/build_train_set.py
```

## Results summary

| Model | Exact-match | Latency (median) |
|---|---|---|
| Base (zero-shot) | 0% | 2.3s |
| Fine-tuned LoRA (PyTorch, unbatched) | 93.0% (57 examples) | ~6.4s |
| Fine-tuned LoRA, GGUF Q8_0 (Vulkan GPU) | 94.7% | **0.26s** |
| Fine-tuned LoRA, GGUF Q8_0 (CPU) | 93.0% | 0.84s |

Full breakdown, methodology, and gotchas (tokenizer vocab fixup, GPU pstate dispatch
bottleneck, batching numerics, etc.) are in `PROGRESS.md`.

## Repo layout

```
data/       eval_set.jsonl, train_set.jsonl
scripts/    dataset generation, fine-tuning, evaluation, GGUF benchmarking
results/    per-run evaluation JSON (accuracy + latency)
```

`checkpoints/`, `gguf/`, and `venv/` are gitignored (large binaries — regenerate locally
via the steps above).
