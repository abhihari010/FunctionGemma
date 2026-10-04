"""Step 5: LoRA fine-tune FunctionGemma 270M on data/train_set.jsonl only.
Trains on the exact tagged function-call format the base model natively
emits (see scripts/schema.py), with loss masked to the completion span."""
import json

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

from schema import FUNCTION_SCHEMA, FIELDS, build_messages

MODEL_ID = "google/functiongemma-270m-it"
OUTPUT_DIR = "checkpoints/functiongemma-270m-lora-intent"
TARGET_STEPS = 504


def format_target(expected):
    parts = ",".join(f"{f}:<escape>{expected[f]}<escape>" for f in FIELDS)
    return f"<start_function_call>call:set_leader_intent{{{parts}}}<end_function_call><end_of_turn>\n"


class IntentDataset(Dataset):
    def __init__(self, rows, tokenizer, processor):
        self.examples = []
        for row in rows:
            prompt_text = processor.apply_chat_template(
                build_messages(row["text"]), tools=[FUNCTION_SCHEMA],
                add_generation_prompt=True, tokenize=False,
            )
            completion_text = format_target(row["expected"])
            full_text = prompt_text + completion_text

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

            labels = list(full_ids)
            labels[: len(prompt_ids)] = [-100] * len(prompt_ids)

            self.examples.append({"input_ids": full_ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, pad_token_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, attention_mask, labels = [], [], []
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append([1] * len(b["input_ids"]) + [0] * pad_len)
        labels.append(b["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    with open("data/train_set.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    dataset = IntentDataset(rows, tokenizer, processor)

    # The step budget is now set DIRECTLY instead of via epochs. Epochs were only ever a
    # proxy for optimizer steps, and the proxy broke twice: round 8's update_only merge shrank
    # the set 1721 -> 1349 rows and silently cut steps 430 -> 337 (v10 trained there, 22%
    # under budget), and round 10's containment data grew it to 1657, which at 3 epochs would
    # have been 621 -- 23% OVER. Neither row-count change touched num_train_epochs because
    # nothing recomputed it. max_steps makes the budget the thing you set, so a data change
    # can no longer move it by accident, and v13 is step-for-step comparable to v11's 504.
    epochs = TARGET_STEPS * 8 / len(rows)
    print(f"{len(rows)} rows, batch 8, {TARGET_STEPS} optimizer steps = {epochs:.2f} epochs")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        # round 4 rebalanced the set by oversampling, 504 distinct rows -> 1511. At 6
        # epochs that is ~4x the gradient exposure that already drove loss to 2e-5, i.e.
        # memorising the duplicates rather than learning the new class balance. 2 epochs
        # holds total optimizer steps near the 288 that converged before (~378 now), so
        # what changes between runs is the class mix, not the amount of training.
        # round 5 dropped avoid_objects from the schema, so less oversampling is needed to
        # balance -- 1511 rows -> 828. 4 epochs keeps optimizer steps near the ~378 that
        # converged in round 4 (828/8 * 4 = 414), so what changes between runs is the
        # schema, not the amount of training.
        # round 6 grew the set to 1282 rows (655 distinct) with the new actions. 3 epochs
        # keeps optimizer steps near the ~414 that converged in round 5 (1282/8*3 = 481).
        # round 7 grew the set to 1721 rows (821 distinct). 2 epochs keeps optimizer steps
        # near the ~481 that converged in round 6 (1721/8*2 = 430).
        # round 8's update_only merge SHRANK the set, 1721 rows -> 1349, so the 2 epochs
        # set for round 7 silently became 337 steps -- 22% under the ~430 target, and v10
        # was trained there. 3 epochs puts it back on budget (1349/8*3 = 506). Watch this
        # whenever the row count moves: the epoch count is a proxy for optimizer steps,
        # and nothing recomputes it. ponytail: a printed step count beats an assert here,
        # the number needs eyeballing against the comment, not just a floor.
        max_steps=TARGET_STEPS,
        # 8x1 and 4x2 both died at step 1 with a raw "CUDA error: out of memory" from the
        # driver (not torch's allocator) while nvidia-smi showed 7.4 GB free. On Windows WDDM
        # a GPU allocation is backed by system RAM, and this machine was down to ~5.5 GB of
        # 16.8 GB free, so the driver refused it. Measured peak at 1x8 is 2.17 GB and it runs
        # clean. Effective batch is still 8, so optimizer steps and the training math are
        # identical to rounds 3-4; only peak activation memory changed.
        # If this ever OOMs again, free system RAM before lowering the effective batch.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
    )
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"saved LoRA adapter to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
