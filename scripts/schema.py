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
                    "enum": ["avoid_regions", "none"],
                    # avoid_objects was removed from this enum: rules doc section 2.2 requires
                    # that "All other objects on the mission field shall be avoided" on every
                    # run, so it is a standing constraint, not something a judge can vary. The
                    # model was spending its hardest decision on a value that is always true --
                    # 22 of its 28 remaining field errors were the avoid_objects/none boundary,
                    # and six different models in the cross-model benchmark missed the same
                    # sentences. Object avoidance is now the autonomy stack's invariant; this
                    # field only reports whether a REGION was placed off limits.
                    "description": "Region-avoidance constraint stated in the command. Object avoidance is always required and is not reported here.",
                },
                "target_location": {
                    "type": "string",
                    "enum": ["NW", "NE", "SW", "SE", "unspecified"],
                    "description": "Field quadrant of the target, or unspecified if not stated.",
                },
                "action": {
                    "type": "string",
                    "enum": [
                        "collect_target", "read_chip", "abort", "return_to_start",
                        "report_status", "pause", "resume",
                        "retry_read", "retry_send",
                    ],
                    # "retrieve" was renamed to "collect_target". The proposed rename was
                    # retrieve_target/retrieve_message, but the one error left in 282 clean
                    # examples is read_chip misread as retrieve -- giving those two a shared
                    # "retrieve_" prefix adds token overlap to exactly the pair that already
                    # confuses. collect_target moves them apart instead.
                    #
                    # BOUNDARY RULES, because several of these compete for the same wording:
                    #   pause vs abort  -- pause expects to resume ("hold", "wait", "stand by",
                    #     "hold one"); abort ends the run ("scrub", "we're done", "stay hidden").
                    #     When a command only says "stop", the presence of concealment or
                    #     finality language decides it, otherwise it is pause.
                    #   A POSITION CORRECTION IS collect_target. Round 8 removed
                    #     update_only: all 245 of its examples corrected the location of the
                    #     target already being pursued, it never carried a constraint, and the
                    #     state machine merges a partial update onto the active task anyway, so
                    #     the action field never needed to encode it. Emit the fields the
                    #     correction states and leave the rest "unspecified"; folding the label
                    #     in lifted v9 from 95.94% to 97.00% across all five sets with no
                    #     retraining. A volunteered fact about a NON-target object would need a
                    #     separate flag, not an action slot -- no such example exists yet.
                    #   retry_read/retry_send -- explicit repetition of a chip read or a
                    #     transmit ("try that scan again", "resend the code").
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
