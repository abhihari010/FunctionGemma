"""Merge the LoRA adapter into the base weights and write an fp16 HF checkpoint that
convert_hf_to_gguf.py will accept.

Was done by hand for the round-2 model; scripted here so a rebuild after any retrain is one
command and the vocab fixup cannot be forgotten.

The fixup, step 2 of PROGRESS.md's reproduction notes: the Gemma3 tokenizer defines 262146
tokens but this text-only 270M model's embedding has 262144 rows, so the converter dies on
`assert max(tokenizer.vocab.values()) < vocab_size`. The two extras are <image_soft_token>
(262144) and <end_of_image> (262145) -- leftover multimodal tokens with no embeddings. Pad
the embedding with two zero rows and set vocab_size to match. Zero rows give logit 0 and
never win greedy decode, so the tokenizer stays byte-identical to the one the PyTorch eval
used, which is what makes the two backends comparable.
"""
import argparse
import json

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID = "google/functiongemma-270m-it"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="checkpoints/functiongemma-270m-lora-intent")
    parser.add_argument("--out", default="checkpoints/merged-fp16")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()

    # pad the embedding to whatever the tokenizer actually declares
    declared = max(tokenizer.get_vocab().values()) + 1
    rows = merged.get_input_embeddings().weight.shape[0]
    if declared > rows:
        print(f"padding embedding {rows} -> {declared} with {declared - rows} zero row(s)")
        merged.resize_token_embeddings(declared)
        with torch.no_grad():
            merged.get_input_embeddings().weight[rows:].zero_()
            out = merged.get_output_embeddings()
            # Gemma ties input/output embeddings; only zero the head if it is a separate tensor
            if out is not None and out.weight.data_ptr() != merged.get_input_embeddings().weight.data_ptr():
                out.weight[rows:].zero_()
    else:
        print(f"no padding needed (embedding {rows}, tokenizer declares {declared})")

    merged.save_pretrained(args.out, safe_serialization=True)
    processor.save_pretrained(args.out)

    # resize_token_embeddings updates config.vocab_size, but assert it rather than trust it --
    # a mismatch here is exactly what kills the converter
    cfg_path = f"{args.out}/config.json"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("vocab_size") != declared:
        cfg["vocab_size"] = declared
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"corrected config.json vocab_size -> {declared}")

    print(f"merged {args.adapter} -> {args.out} (vocab_size={declared})")


if __name__ == "__main__":
    main()
