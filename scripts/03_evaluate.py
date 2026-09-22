"""Step 6: evaluate a model (base or LoRA fine-tuned) against the locked
data/eval_set.jsonl. Reports exact-match + per-field accuracy and raw
PyTorch inference latency. Never edits eval_set.jsonl -- read only."""
import argparse
import json
import time

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

from schema import FUNCTION_SCHEMA, FIELDS, build_messages, parse_function_call

MODEL_ID = "google/functiongemma-270m-it"


def load_eval_set(path="data/eval_set.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="path to a LoRA adapter dir; omit for base model")
    parser.add_argument("--label", default=None, help="name for this run in the results file")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="prompts per generate() call. 1 = true single-request latency "
                             "(~7s/example on this laptop); 16 = same accuracy, ~10x faster wall clock")
    args = parser.parse_args()

    label = args.label or (args.adapter if args.adapter else "base-zero-shot")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.padding_side = "left"  # right padding would make generate() continue from pad tokens
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()

    eval_rows = load_eval_set()
    n = len(eval_rows)

    def sync():
        if model.device.type == "cuda":
            torch.cuda.synchronize()

    prompts = [
        processor.apply_chat_template(
            build_messages(row["text"]), tools=[FUNCTION_SCHEMA],
            add_generation_prompt=True, tokenize=False,
        )
        for row in eval_rows
    ]
    prompt_lens = [len(tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts]

    # ponytail: sort by length so each batch pads as little as possible. Padding is not free --
    # it perturbs bf16 attention numerics enough to flip borderline examples (measured on this
    # eval set: bs=1 and length-sorted bs=16 both score 53/57, unsorted bs=16 scores 51/57,
    # one-big-batch bs=57 scores 52/57). Prompts here span only 292-310 tokens; if the eval set
    # ever gets a wider length spread, shrink --batch-size rather than trusting the sort alone.
    order = sorted(range(n), key=lambda i: prompt_lens[i])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else processor.eos_token_id

    parsed_all = [None] * n
    latencies = [None] * n

    sync()
    wall_t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            idx = order[start:start + args.batch_size]
            batch = tokenizer(
                [prompts[i] for i in idx], return_tensors="pt",
                padding=True, add_special_tokens=False,
            ).to(model.device)

            sync()
            t0 = time.perf_counter()
            out = model.generate(
                **batch, pad_token_id=pad_id, max_new_tokens=args.max_new_tokens,
                do_sample=False,  # model's default generation_config samples (do_sample=True);
                                  # greedy decoding is needed for a reproducible accuracy/latency benchmark
            )
            sync()
            per_example_s = (time.perf_counter() - t0) / len(idx)

            prompt_width = batch["input_ids"].shape[-1]
            for i, seq in zip(idx, out):
                parsed_all[i] = parse_function_call(
                    tokenizer.decode(seq[prompt_width:], skip_special_tokens=True)
                )
                latencies[i] = per_example_s
    wall_clock_s = time.perf_counter() - wall_t0

    field_correct = {f: 0 for f in FIELDS}
    exact_correct = 0
    per_example = []
    for row, parsed, lat in zip(eval_rows, parsed_all, latencies):
        expected = row["expected"]
        row_correct = {f: parsed[f] == expected[f] for f in FIELDS}
        for f in FIELDS:
            field_correct[f] += row_correct[f]
        is_exact = all(row_correct.values())
        exact_correct += is_exact
        per_example.append({
            "text": row["text"], "expected": expected, "parsed": parsed,
            "exact_match": is_exact, "latency_s": lat,
        })

    def accuracy_by(key):
        groups = {}
        for row, ex in zip(eval_rows, per_example):
            hits, total = groups.get(key(row), (0, 0))
            groups[key(row)] = (hits + ex["exact_match"], total + 1)
        return {k: {"exact_match": hits / total, "n": total}
                for k, (hits, total) in sorted(groups.items())}

    summary = {
        "label": label,
        "n_examples": n,
        "exact_match_accuracy": exact_correct / n,
        "per_field_accuracy": {f: field_correct[f] / n for f in FIELDS},
        "batch_size": args.batch_size,
        "wall_clock_s": wall_clock_s,
        # accuracy sliced two ways: by eval_set.jsonl's "set" tag (core57 = the frozen
        # original 57, ext = later coverage additions) and by expected action, because a
        # headline average hides a weak class -- read_chip once sat at 2/3 unnoticed
        "by_set": accuracy_by(lambda row: row.get("set", "all")),
        "by_action": accuracy_by(lambda row: row["expected"]["action"]),
        # at --batch-size 1 this is true per-request latency; above that it is batch time
        # amortised per example (throughput), which is NOT a latency number for the report
        "latency_s": {
            "mean": sum(latencies) / n,
            "median": sorted(latencies)[n // 2],
            "min": min(latencies),
            "max": max(latencies),
        },
    }

    print(json.dumps(summary, indent=2))

    import os
    os.makedirs("results", exist_ok=True)
    out_path = f"results/{label.replace('/', '_')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "examples": per_example}, f, indent=2)
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
