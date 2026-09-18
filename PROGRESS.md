# FunctionGemma Intent Parser — Progress Log

Raytheon AVC "Mission Bird Dog", D-010: build eval set, fine-tune FunctionGemma 270M,
measure schema accuracy + latency for the Sep 30 report.

## Status: Steps 1-6 (raw PyTorch) done. Quantized/GGUF latency not yet done.

## Environment (Step 1) — done
- Repo root: `C:\Users\abhih\FunctionGemma`
- venv at `venv/` — activate via `./venv/Scripts/python.exe` (or `venv\Scripts\activate` in PowerShell)
- `requirements.txt` has the frozen package list (torch 2.6.0+cu124, transformers 5.17.0, peft 0.21.0, trl 1.13.0, accelerate, datasets)
- GPU confirmed: NVIDIA RTX 4070 Laptop GPU, 8GB VRAM, `torch.cuda.is_available()` = True
- Model: **`google/functiongemma-270m-it`** (HF/PyTorch trainable build, confirmed via model card — gated by Google's license)
  - HF login done, token cached in venv's HF credential store under account `abhihari010`. If it expires, re-run login (see "Gotchas" below).

## Schema decisions
- Base schema from the task: `target_color` (blue/red/yellow/black), `constraints` (avoid_objects/avoid_regions/none), `target_location` (NW/NE/SW/SE/unspecified), `action` (read_chip/retrieve/abort/return_to_start).
- **Extended `target_color` with a 5th value `"unspecified"`** — decided with the user. Several canonical Leader's Intent examples (abort, avoid-region-only) name no color, and the original 4-value enum had no fallback (unlike `target_location`, which already had `unspecified`).
- Two canonical section-2.2 example sentences use placeholder colors **not in the competition's actual palette** (section 2.7 lists only blue/red/yellow/black): "green box" and "orange box". Since the doc calls the object itself a placeholder ("box is used just as a placeholder"), the color words were swapped to valid ones (green→black, orange→blue) rather than injecting invalid classes into the data. Documented inline in `scripts/build_eval_set.py`.
- **Fixed 2026-09-17 (caught by user review):** my own invented paraphrases (not the verbatim section-2.2 quotes) had been calling the target object a "box"/"cube" in several sentences, but section 2.7 explicitly says the real Target Objects are **6-inch foam dice**. Swapped those invented instances to "die" in both `build_eval_set.py` (7 sentences) and `build_train_set.py` (3 templates). The verbatim section-2.2 quotes still say "box" on purpose — that's literally what the rules doc says, using "box" as placeholder phrasing per its own footnote — only my own inventions were wrong.
- **Follow-up fix, same day:** after the above fix, `train_set.jsonl` had zero "box"/"cube" examples (all "die"), while the eval set still has 5 verbatim "box" sentences from the rules doc. Training exclusively on "die" would leave the model narrower than needed — a real judge is just as likely to use the rules doc's own "box" wording, or say "cube"/"block" informally. Added a `NOUNS = ["die", "box", "cube", "block"]` rotation across the same 3 templates in `build_train_set.py` so the model sees the object-noun vary (12 examples total: 3 each). Data regenerated (still 0 train/eval overlap), model retrained and re-evaluated — numbers below reflect this final version of the data.
- Colorless/action-implicit utterances (e.g. "Avoid the southwest quarter of the field.") were labeled `action: retrieve` on the assumption the background retrieval mission continues; this is an inference call, not stated explicitly in the doc — worth double-checking if it materially matters for the report.

## FunctionGemma output format (found empirically, not JSON as first assumed)
The model does NOT emit plain JSON. It emits its own tagged syntax:
```
<start_function_call>call:set_leader_intent{target_color:<escape>red<escape>,constraints:<escape>none<escape>,...}<end_function_call>
```
Parsing/formatting logic lives in `scripts/schema.py` (`parse_function_call`, `build_messages`, `FUNCTION_SCHEMA`). Chat template requires a `developer`-role message (not `system`) to activate function-calling behavior; roles are `developer` / `user` / `model` (Gemma-style), turns end with `<end_of_turn>`. Generation stop tokens: `[1 (<eos>), 50, 106 (<end_of_turn>)]`.

## Data (Steps 3 & 4) — done, locked
- `data/eval_set.jsonl` — 57 examples: all canonical Leader's Intent phrasings from rules doc section 2.2 (9 sentences, colors substituted per above) + 12 invented paraphrases per category (target retrieval, abort, avoid-region, object-relocated). Built once via `scripts/build_eval_set.py` — **do not re-run/edit this**, per the test/train separation rule.
- `data/train_set.jsonl` — 140 examples, template x slot-value generated via `scripts/build_train_set.py`, verified **zero text overlap** with eval_set.jsonl (script asserts this before writing).
- Source syllabus: `Syllabus/2627 Raytheon AVC Rules v0.3 (1).pdf`, section 2.2 (pages 6-8) has the Leader's Intent examples; section 2.7 (page 12) confirms the 4 real target colors.

## Fine-tuning (Step 5) — done
- `scripts/02_finetune_lora.py` — LoRA via PEFT (`r=16, alpha=32, target_modules="all-linear"`), plain HF `Trainer` with a hand-rolled dataset/collator (prompt tokens masked with `-100`, loss only on the completion span). Chose this over `trl.SFTTrainer` because the non-standard `developer` role + custom tag-format completions didn't map cleanly onto SFTTrainer's chat-template assumptions — a manual masked-CLM loop was more predictable.
- 6 epochs, batch size 8, lr 2e-4, ~7.5 min on the RTX 4070. Loss went 0.53 → ~0.00003 (clean convergence, small/templated dataset).
- Adapter saved to `checkpoints/functiongemma-270m-lora-intent/`.

## Evaluation (Step 6, raw PyTorch) — done
Script: `scripts/03_evaluate.py --adapter <path> --label <name>` (omit `--adapter` for zero-shot base). Results saved to `results/<label>.json` (summary + every example's parsed vs expected + per-example latency).

| Model | Exact-match | color | constraints | location | action | Latency (mean, unbatched greedy, RTX 4070) |
|---|---|---|---|---|---|---|
| Base (zero-shot) | **0%** (0/57) | 71.9% | 26.3% | 35.1% | 10.5% | 2.29s |
| Fine-tuned LoRA | **93.0%** (53/57) | 100% | 98.2% | 98.2% | 96.5% | 7.39s* |

(These are from the final data — after both the "die" fix and the noun-diversity follow-up fix — training/eval both re-run each time. Accuracy has stayed consistently in the 91-93% exact-match range across all three data revisions, so none of the terminology fixes were materially skewing the headline result.)

## Latency re-benchmark (2026-09-17, later same day) — real bug found + a hardware caveat

Investigated the latency-variance flag above. Found a real bug:

- **Bug:** `model.generate()` was never passed `do_sample`, so it fell back to the model's
  default `generation_config` (`do_sample: true, top_k: 64, top_p: 0.95`) — every eval run was
  *sampling*, not doing deterministic greedy decoding. This explains both the accuracy drift
  across runs (91.2% / 93.0% / 93.0%) and part of the latency variance (sampled sequences run
  different lengths before hitting a stop token).
- **Fix:** added `do_sample=False` explicitly in both `scripts/03_evaluate.py` and
  `scripts/00_zeroshot_baseline.py`. Re-ran both baseline and fine-tuned eval — accuracy is now
  reproducible: base 0.0% exact-match, fine-tuned **93.0%** (per-field: color 98.2%, constraints
  100%, location 98.2%, action 96.5%). This is the accuracy number to use going forward.

Latency was *still* variable after the fix (min 1.9s, mean 5.4s, max 8.6s), and identically so
on the base model — ruled out sampling as the (sole) cause. Tried a 3-call GPU warm-up before
timing on the theory that the laptop GPU downclocks to its idle P8 state between calls and pays
a ramp-up cost per call; **this made it worse** (min jumped to 5.7s, everything clustered ~7s) —
disproving the ramp-up theory and instead pointing at **thermal throttling from several
back-to-back full benchmark runs in one session** (each run is ~10+ min of sustained GPU compute
on a laptop chassis; nvidia-smi showed 63°C at idle even between runs). Reverted the warm-up
(it didn't help, so it's not staying in the script).

**Conclusion:** the `do_sample=False` fix is real and should stay — it makes accuracy
reproducible. The *latency* number is currently unreliable because this session ran the
benchmark back-to-back many times in a row; the ~1.9s minimum observed across every run (cold or
not) is the most trustworthy floor. **For the report, re-run
`python scripts/03_evaluate.py --adapter checkpoints/functiongemma-270m-lora-intent --label finetuned-lora`
once, cold (GPU idle, first run of the session)**, and use that number rather than anything from
this session's later, thermally-compounded runs.

**Update:** ran it cold (fresh terminal, GPU at idle beforehand) — still got mean 6.58s / median
6.75s / min 2.29s / max 7.52s (`results/finetuned-lora.json`), not the ~1.9s from the very first
run of the session. `nvidia-smi` post-run shows no throttle flags (idle-only), so the
thermal-throttling theory doesn't fully explain it either. **Honest conclusion: unbatched raw
PyTorch latency on this laptop genuinely varies run-to-run, roughly in the 2-7.5s band, and I
have not root-caused why the very first run was an outlier at ~1.9s.** Accuracy (93.0%) is solid
and reproducible; for the report, quote latency as a range (min ~2s, typical ~6-7s) rather than a
single number, and note it's unbatched single-example greedy decode on a laptop GPU — the
quantized/GGUF measurement (not yet done) is the more deployment-representative number anyway.

- Latency is essentially unchanged base→fine-tuned, as expected (LoRA doesn't change inference cost; both use `max_new_tokens=128` greedy decode).
- The 4 remaining failures (`results/finetuned-lora.json`) are all single-field slips on genuinely ambiguous phrasings, e.g. "return the yellow box to the start" parsed as `action: return_to_start` instead of `retrieve`.
- This before/after comparison is the headline number for the Sep 30 report.

## Latency root-caused (2026-09-17, session 3) — it was never thermal or sampling

The open question above ("I have not root-caused why the very first run was an outlier") is
resolved. **The eval was dispatch-bound on a GPU that never leaves a low power state.** Evidence:

- A **1-token** decode forward costs **144 ms** — the same as a **303-token** prefill forward
  (164 ms). Cost independent of work done means the time is per-kernel overhead, not compute:
  ~2,200 tiny kernel launches per forward, each paying Windows/WDDM dispatch cost.
- `nvidia-smi` sampled *during* the eval (not after, which is why earlier sessions missed it):
  SM clock **537-660 MHz against a 3105 MHz max**, 11-12 W, 25-34% utilisation, pstate **P4/P5**.
  The work is too small and too serial for the driver to ever promote the GPU to P0.
- The GPU itself is fine: a 4096^3 bf16 matmul measures **35 TFLOPS** on the same process.

This also explains the 2-7.5 s run-to-run band: latency depends entirely on which pstate the GPU
happens to be in when a run starts. Thermal throttling was ruled out (no throttle flags, 11 W
draw); the warm-up experiment "making it worse" is consistent with pstate luck, not ramp-up cost.

**Fix applied:** `scripts/03_evaluate.py` now batches with `--batch-size` (default 16),
length-sorted so each batch pads as little as possible. Full eval wall clock **295 s -> 30 s
(9.7x)** with **identical accuracy (53/57 = 93.0%, per-field unchanged)**.

Batching is not numerically free, which is why the default is 16 and not 57:

| config | wall clock | exact-match |
|---|---|---|
| `--batch-size 1` (old behaviour) | 295 s | 53/57 = 93.0% |
| `--batch-size 8` | 58 s | 53/57 = 93.0% |
| **`--batch-size 16` (default)** | **28-30 s** | **53/57 = 93.0%** |
| `--batch-size 57` (one batch) | 8 s | 52/57 = 91.2% |
| unsorted `--batch-size 16` | 28 s | 51/57 = 89.5% |

Padding perturbs bf16 attention numerics enough to flip genuinely borderline examples. Useful
corollary for the report: **the 93.0% headline carries about +/-1 example (~2%) of numerical
noise** — which independently explains the "91-93% across all three data revisions" band noted
above. The 4 stable misses are the same ones listed below; the 5th that appears at bs=57 is
"Forget the target, just bring the system back to the start".

**Latency for the report:** quote `--batch-size 1` (true single-request latency, ~5-7 s/example)
and state it is unbatched HF-eager PyTorch on a laptop GPU pinned in P4/P5 — an artifact of the
harness, not of the 270M model. Do **not** quote the batched per-example number as latency; it is
throughput (the JSON now records `batch_size` and `wall_clock_s` so the two can't be confused).

**Two faster paths exist but are blocked on this machine:**
- `torch.compile(mode="reduce-overhead")` + static cache (CUDA graphs) would collapse the launch
  overhead — **blocked: no Triton for Windows in stock PyTorch 2.6** (needs the `triton-windows`
  package). This is the real single-request fix if latency ever needs to drop for its own sake.
- Locking clocks via `nvidia-smi -lgc` — **blocked: requires admin** on this machine.

Neither matters much for the deliverable: the GGUF/llama.cpp number below is the
deployment-representative latency anyway, and llama.cpp launches far fewer, fused kernels.

## GGUF / llama.cpp benchmark (2026-09-17, session 3) — DONE

Step 6's deployment number is measured. **Q8_0 GGUF on the GPU is 25x faster than raw PyTorch on
the same GPU, at identical accuracy.**

| build | backend | median latency | exact-match | decode tok/s | prompt eval |
|---|---|---|---|---|---|
| PyTorch bf16 (`--batch-size 1`) | RTX 4070 | 6.39 s | 53/57 = 93.0% | ~6.6 | - |
| **GGUF Q8_0** | **RTX 4070 (Vulkan)** | **0.256 s** | **54/57 = 94.7%** | **195** | 34 ms |
| GGUF Q8_0 | CPU (14 threads) | 0.838 s | 53/57 = 93.0% | 88 | 361 ms |
| GGUF Q4_K_M | RTX 4070 (Vulkan) | 0.299 s | 50/57 = 87.7% | 166 | 40 ms |

Headline for the report: **~0.26 s/command on the laptop GPU, ~0.84 s CPU-only**, versus 6.4 s for
the same weights under HF PyTorch. Even **CPU-only GGUF beats the PyTorch GPU path by 7.6x** —
which confirms the session-3 finding above that the 6-7 s was harness overhead, not the model.

**Use Q8_0, not Q4_K_M.** Q4_K_M loses 3 more examples (87.7%, and it degrades `constraints` and
`action`, the two fields the fine-tune actually fixed) while being *slower* than Q8_0 — at 270M the
dequant cost outweighs the smaller memory footprint. Q8_0 is 286 MB, Q4_K_M 250 MB; the 36 MB is
not worth 3 examples. (Both are dominated by the 262k-row embedding, which is why Q4 saves so
little.)

Accuracy note: Q8_0/Vulkan scores 54/57 vs PyTorch's 53/57 — it gets "Priority target is red..."
right. That is **not** a real improvement, it is the same +/-1 numerical noise documented above;
the other 3 misses are identical in both. CPU Q8_0 lands on exactly PyTorch's 53/57 with exactly
the same 4 misses, which is the cleanest evidence the conversion is faithful.

### How it was produced (reproducible)

1. Merge LoRA -> fp16 HF checkpoint at `checkpoints/merged-fp16` (`merge_and_unload()`, save_pretrained).
2. **Vocab fixup, required:** the Gemma3 tokenizer defines 262146 tokens but this text-only 270M
   model's embedding has 262144 rows, so `convert_hf_to_gguf.py` dies on
   `assert max(tokenizer.vocab.values()) < vocab_size`. The two extras are `<image_soft_token>`
   (262144) and `<end_of_image>` (262145) — leftover multimodal tokens with no embeddings. Fix:
   pad `model.embed_tokens.weight` with 2 zero rows and set `config.json` `vocab_size` to 262146.
   Zero rows give logit 0 and never win greedy decode; this keeps the tokenizer byte-identical to
   the one the PyTorch eval used, which matters for a fair comparison.
3. `python llama.cpp/convert_hf_to_gguf.py checkpoints/merged-fp16 --outfile gguf/...-f16.gguf --outtype f16`
4. `llama-quantize.exe gguf/...-f16.gguf gguf/...-q8_0.gguf Q8_0`
5. `llama-server.exe -m gguf/...-q8_0.gguf --port 8081 -c 4096 -np 1 -ngl 99`
6. `python scripts/04_benchmark_gguf.py --port 8081 --label gguf-vulkan-q8_0`

**The custom tokens survived conversion** (the risk flagged earlier): `<start_function_call>`=48,
`<end_function_call>`=49, `<escape>`=52, `<end_of_turn>`=106, arch `gemma3`, chat template embedded.

Two gotchas worth keeping:
- GGUF metadata only carries `eos_token_id=1`, but HF stopped on `[1, 50, 106]`. The fine-tune ends
  its call with `<end_of_turn>` (106), so `04_benchmark_gguf.py` passes `"stop": ["<end_of_turn>"]`
  explicitly. Without it generation runs to `n_predict` and the latency number is meaningless.
- The benchmark sends **token ids**, not text, so llama.cpp and PyTorch see byte-identical prompts.
- `cache_prompt` (reusing the shared tool-schema prefix) is a wash here: prompt eval is only 34 ms
  of the 256 ms, decode dominates. Left off by default so the number stays comparable.

Binaries used: llama.cpp `b11026` prebuilt Windows CPU + Vulkan builds (no CUDA toolkit or compiler
on this machine; Vulkan reaches the 4070 without the 400 MB CUDA runtime download). Artifacts in
`gguf/`, results in `results/gguf-*.json`.

## Not yet done
- ~~**Quantized/GGUF latency**~~ — **DONE, see the GGUF section above** (0.256 s/command, Q8_0 on GPU). Original plan kept below for reference. Rough plan when resuming:
  1. Merge LoRA into base weights (already done in-memory in `03_evaluate.py` via `merge_and_unload()`; for GGUF conversion, save the merged model to disk with `model.save_pretrained(...)`).
  2. Get llama.cpp (prebuilt Windows release or build from source) + its `convert_hf_to_gguf.py`.
  3. Convert merged model → GGUF, quantize (e.g. Q8_0 — there's already a community `ggml-org/functiongemma-270m-it-GGUF` for the *base* model on HF that could be a useful reference/sanity-check for the conversion recipe, but our fine-tune needs its own conversion since it's a custom adapter).
  4. Benchmark via `llama-cli`/`llama-bench` or Ollama, comparable latency methodology to `03_evaluate.py` (per-example wall time, same eval sentences).
  5. Watch for custom special tokens (`<start_function_call>`, `<escape>`, `<end_function_call>`) surviving the GGUF conversion/tokenizer correctly.
  - Estimated effort: 10-20 min if clean, up to ~30 min if compatibility snags appear.
- Final Sep 30 report writeup (short summary doc) — not started; all the numbers above are ready to drop in once GGUF latency is either measured or explicitly deferred.

## Gotchas / notes for next session
- Windows console + `huggingface-cli`/`hf` CLI banner text crashes with `UnicodeEncodeError` (cp1252 can't encode emoji) — avoid the CLI, use `huggingface_hub.login(token=...)` from Python instead, or set `HF_TOKEN` env var.
- `pip install torch` alone gives CPU-only on Windows; need `--index-url https://download.pytorch.org/whl/cu124` for the CUDA build.
- Run scripts from the repo root (`C:\Users\abhih\FunctionGemma`), not from `scripts/` — they use relative paths like `data/eval_set.jsonl`.
