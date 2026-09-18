"""Step 6 (deployment): benchmark a GGUF build of the intent parser against the
locked data/eval_set.jsonl, via a running llama-server.

Sends the SAME token ids the PyTorch eval uses (rendered by the HF chat template,
then tokenized), so prompts are byte-identical across backends and the accuracy /
latency comparison against results/finetuned-lora.json is apples-to-apples.

Start a server first, e.g.:
  llama-server.exe -m gguf/functiongemma-270m-intent-q8_0.gguf --port 8081 -c 4096 -np 1 -ngl 99
then:
  python scripts/04_benchmark_gguf.py --port 8081 --label gguf-vulkan-q8_0
"""
import argparse
import json
import statistics
import time
import urllib.request

from transformers import AutoProcessor

from schema import FUNCTION_SCHEMA, FIELDS, build_messages, parse_function_call

MODEL_ID = "google/functiongemma-270m-it"


def build_prompts():
    """Render + tokenize with the HF tokenizer so llama.cpp gets identical ids."""
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    with open("data/eval_set.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for row in rows:
        text = processor.apply_chat_template(
            build_messages(row["text"]), tools=[FUNCTION_SCHEMA],
            add_generation_prompt=True, tokenize=False,
        )
        row["ids"] = tokenizer(text, add_special_tokens=False)["input_ids"]
    return rows


def completion(port, ids, max_new_tokens, cache_prompt):
    payload = {
        "prompt": ids, "n_predict": max_new_tokens,
        "temperature": 0.0, "top_k": 1,  # greedy, to match the PyTorch benchmark
        "cache_prompt": cache_prompt,
        # GGUF metadata only carries eos=1; the fine-tune ends its call with
        # <end_of_turn>, so stop on it explicitly or generation runs to n_predict
        "stop": ["<end_of_turn>"],
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--label", default="gguf")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--cache-prompt", action="store_true",
                        help="let llama-server reuse the shared tool-schema prefix "
                             "(deployment-realistic; off by default so the number is "
                             "comparable to the PyTorch eval, which re-prefills every time)")
    args = parser.parse_args()

    rows = build_prompts()
    completion(args.port, rows[0]["ids"], args.max_new_tokens, args.cache_prompt)  # warm

    latencies, n_tokens, predicted_ms, prompt_ms, parsed_all = [], [], [], [], []
    for row in rows:
        t0 = time.perf_counter()
        res = completion(args.port, row["ids"], args.max_new_tokens, args.cache_prompt)
        latencies.append(time.perf_counter() - t0)
        timings = res.get("timings", {})
        n_tokens.append(timings.get("predicted_n", 0))
        predicted_ms.append(timings.get("predicted_ms", 0.0))
        prompt_ms.append(timings.get("prompt_ms", 0.0))
        parsed_all.append(parse_function_call(res.get("content", "")))

    n = len(rows)
    exact = sum(all(p[f] == r["expected"][f] for f in FIELDS) for p, r in zip(parsed_all, rows))
    summary = {
        "label": args.label,
        "n_examples": n,
        "cache_prompt": args.cache_prompt,
        "exact_match_accuracy": exact / n,
        "per_field_accuracy": {
            f: sum(p[f] == r["expected"][f] for p, r in zip(parsed_all, rows)) / n
            for f in FIELDS
        },
        "latency_s": {
            "mean": statistics.mean(latencies), "median": statistics.median(latencies),
            "min": min(latencies), "max": max(latencies),
        },
        "tokens_generated": {"mean": statistics.mean(n_tokens), "total": sum(n_tokens)},
        "decode_tok_per_s": sum(n_tokens) / (sum(predicted_ms) / 1000) if sum(predicted_ms) else None,
        "prompt_ms_median": statistics.median(prompt_ms),
        "wall_clock_s": sum(latencies),
    }
    print(json.dumps(summary, indent=2))

    out_path = f"results/{args.label.replace('/', '_')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "examples": [
                {"text": r["text"], "expected": r["expected"], "parsed": p,
                 "exact_match": all(p[f] == r["expected"][f] for f in FIELDS)}
                for p, r in zip(parsed_all, rows)
            ],
        }, f, indent=2)
    print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
