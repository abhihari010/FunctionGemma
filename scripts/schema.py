"""Shared function schema + parsing helpers for the intent parser."""
import json
import re

FUNCTION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "set_leader_intent",
        "description": "Extract the structured Leader's Intent from a judge's spoken/text command during the AVC Mission Bird Dog challenge.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_color": {
                    "type": "string",
                    "enum": ["blue", "red", "yellow", "black", "unspecified"],
                    "description": "Color of the target object, or unspecified if no target object is named.",
                },
                "constraints": {
                    "type": "string",
                    "enum": ["avoid_objects", "avoid_regions", "none"],
                    "description": "Explicit avoidance constraint stated in the command.",
                },
                "target_location": {
                    "type": "string",
                    "enum": ["NW", "NE", "SW", "SE", "unspecified"],
                    "description": "Field quadrant of the target, or unspecified if not stated.",
                },
                "action": {
                    "type": "string",
                    "enum": ["read_chip", "retrieve", "abort", "return_to_start"],
                    "description": "The action the vehicle system should take.",
                },
            },
            "required": ["target_color", "constraints", "target_location", "action"],
        },
    },
}

FIELDS = ("target_color", "constraints", "target_location", "action")
DEVELOPER_MESSAGE = "You are a model that can do function calling with the following functions"


def build_messages(user_text):
    return [
        {"role": "developer", "content": DEVELOPER_MESSAGE},
        {"role": "user", "content": user_text},
    ]


_TAG_PATTERN = re.compile(r"(\w+):<escape>(.*?)<escape>")


def parse_function_call(raw_output):
    """Best-effort extraction of the 4 label fields from raw model text.
    FunctionGemma emits its own tagged syntax, e.g.:
      <start_function_call>call:set_leader_intent{action:<escape>retrieve<escape>,...}<end_function_call>
    rather than plain JSON, so parse that directly. Falls back to a JSON
    object (for {"target_color": ...} / {"arguments": {...}} shapes) in case
    a fine-tuned checkpoint is coaxed into a different format.
    Returns dict with the 4 fields (value None where missing/unparseable)."""
    result = {f: None for f in FIELDS}

    tag_matches = _TAG_PATTERN.findall(raw_output)
    if tag_matches:
        for key, value in tag_matches:
            if key in result:
                result[key] = value
        return result

    match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not match:
        return result
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return result
    args = obj.get("arguments", obj) if isinstance(obj, dict) else {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    for f in FIELDS:
        if f in args:
            result[f] = args[f]
    return result
