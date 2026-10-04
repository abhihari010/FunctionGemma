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



# Exact schema-constrained decoding. The GBNF grammar in gguf/leader_intent.gbnf does this
# for llama.cpp, but the PyTorch eval path ignores it, and v14 paid for that: 8 of its 18
# target_location errors were values that are not even in that field's enum ("N", "S", "E",
# "W", and once "North", where the only legal values are NW/NE/SW/SE/unspecified). No amount
# of training data makes an out-of-enum token impossible; constrained decoding does.
#
# The output is a fixed template with six slots from small enums, so there is no need for a
# grammar engine. Walk the slots left to right and, at each one, score every legal value as a
# continuation of what has been decided so far, then keep the best. That is exact constrained
# argmax -- stronger than the per-token greedy a grammar gives, because it compares whole
# values rather than committing to a first token it cannot take back.
#
# On cost: a row takes 6 forward passes here instead of ~60 autoregressive steps, but this
# loop does ONE ROW AT A TIME (it batches across a slot's candidates, not across rows) while
# the unconstrained path batches 16 prompts per generate(). Measured over all 740 examples
# that is 742 ms/example constrained vs 470 ms/example unconstrained -- so it is SLOWER here,
# not faster. It beats unconstrained at batch 1 (~7 s/example, see the batch-size note below)
# by a wide margin. Batching rows as well would close the gap; not done, because accuracy is
# what this path is for and 742 ms is not the bottleneck.
PREFIX = "<start_function_call>call:set_leader_intent{"
ENUMS = {f: FUNCTION_SCHEMA["function"]["parameters"]["properties"][f]["enum"] for f in FIELDS}


def decode_constrained(model, tokenizer, prompt, device):
    """Return the schema-valid field dict with the highest total log-probability."""
    text = prompt + PREFIX
    chosen = {}
    for slot, field in enumerate(FIELDS):
        text += f"{field}:<escape>"
        base = tokenizer(text, add_special_tokens=False)["input_ids"]
        cands = ENUMS[field]
        cand_ids = [tokenizer(v, add_special_tokens=False)["input_ids"] for v in cands]

        width = max(len(base) + len(c) for c in cand_ids)
        pad = tokenizer.pad_token_id or 0
        rows, spans = [], []
        for c in cand_ids:
            seq = base + c
            rows.append(seq + [pad] * (width - len(seq)))
            spans.append((len(base), len(seq)))
        ids = torch.tensor(rows, device=device)
        mask = torch.tensor([[1] * e + [0] * (width - e) for _, e in spans], device=device)

        # logits_to_keep is load-bearing, not an optimisation. The full tensor here is
        # candidates x ~315 positions x 262144 vocab, which OOMs an 8 GB card at 2.7 GB a
        # slot. Only the candidate span matters, so ask for just the tail.
        keep = width - min(b for b, _ in spans) + 1
        logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=keep).logits
        kept_from = width - logits.shape[1]  # absolute position of logits[:, 0]
        logprobs = torch.log_softmax(logits.float(), dim=-1)

        scores = []
        for r, (b, e) in enumerate(spans):
            # the token at position k is predicted by the logits at k-1
            tok = ids[r, b:e]
            lp = logprobs[r, b - 1 - kept_from:e - 1 - kept_from, :]
            scores.append(lp.gather(-1, tok.unsqueeze(-1)).sum().item())

        best = cands[max(range(len(cands)), key=lambda i: scores[i])]
        chosen[field] = best
        text += best + "<escape>"
        if slot < len(FIELDS) - 1:
            text += ","
    return chosen

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="path to a LoRA adapter dir; omit for base model")
    parser.add_argument("--label", default=None, help="name for this run in the results file")
    # data/heldout_set.jsonl is the fresh set written after eval_set.jsonl had been
    # used for model selection three times and stopped being a clean estimate
    parser.add_argument("--eval-set", default="data/eval_set.jsonl",
                        help="eval set to score against")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--constrain", action="store_true",
                        help="decode under the schema enums (see decode_constrained). "
                             "Makes out-of-enum output impossible; also faster.")
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

    eval_rows = load_eval_set(args.eval_set)
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
        if args.constrain:
            for i in order:
                sync()
                t0 = time.perf_counter()
                parsed_all[i] = decode_constrained(model, tokenizer, prompts[i], model.device)
                sync()
                latencies[i] = time.perf_counter() - t0
        else:
            for begin in range(0, n, args.batch_size):
                idx = order[begin:begin + args.batch_size]
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
