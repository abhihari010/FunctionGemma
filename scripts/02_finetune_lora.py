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

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=6,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
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
