"""Step 2 sanity check: load base FunctionGemma 270M and run a handful of
hand-written sentences through it. Working proof, not the rigorous eval
(that's data/eval_set.jsonl via 03_evaluate.py)."""
import json
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

from schema import FUNCTION_SCHEMA, build_messages, parse_function_call

MODEL_ID = "google/functiongemma-270m-it"

QUICK_SAMPLES = [
    ("Retrieve the red box and bring it back to the starting zone.",
     {"target_color": "red", "constraints": "none", "target_location": "unspecified", "action": "retrieve"}),
    ("Abort the mission, remain undetected.",
     {"target_color": "unspecified", "constraints": "none", "target_location": "unspecified", "action": "abort"}),
    ("Avoid the northeast quarter of the field.",
     {"target_color": "unspecified", "constraints": "avoid_regions", "target_location": "NE", "action": "retrieve"}),
    ("The yellow object has moved to the southwest corner.",
     {"target_color": "yellow", "constraints": "none", "target_location": "SW", "action": "retrieve"}),
    ("Get the blue box and avoid all other objects on the field.",
     {"target_color": "blue", "constraints": "avoid_objects", "target_location": "unspecified", "action": "retrieve"}),
    ("Just scan the chip on the black object, don't bring it back.",
     {"target_color": "black", "constraints": "none", "target_location": "unspecified", "action": "read_chip"}),
    ("Return to the starting zone now, we're done.",
     {"target_color": "unspecified", "constraints": "none", "target_location": "unspecified", "action": "return_to_start"}),
]


def main():
    print(f"loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
    print("loaded. device:", model.device)

    correct = 0
    for text, expected in QUICK_SAMPLES:
        messages = build_messages(text)
        inputs = processor.apply_chat_template(
            messages, tools=[FUNCTION_SCHEMA], add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        out = model.generate(**inputs, pad_token_id=processor.eos_token_id, max_new_tokens=128, do_sample=False)
        raw = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        parsed = parse_function_call(raw)
        is_match = parsed == expected
        correct += is_match
        print("-" * 60)
        print("text:", text)
        print("raw output:", raw.strip())
        print("parsed:", parsed)
        print("expected:", expected)
        print("exact match:", is_match)

    print("=" * 60)
    print(f"quick sanity accuracy: {correct}/{len(QUICK_SAMPLES)}")


if __name__ == "__main__":
    main()
