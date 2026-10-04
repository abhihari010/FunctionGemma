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
                "target_location": {
                    "type": "string",
                    "enum": ["NW", "NE", "SW", "SE", "unspecified"],
                    # v3 narrowed this back to its stated meaning. Through v11 it was
                    # OVERLOADED: all 212 avoid_regions training rows carried the AVOIDED
                    # region here with target_color unspecified, so a consumer reading
                    # target_location=NW could not tell "go there" from "never go there".
                    # Regions now live in constraint_region and this field is the target only.
                    "description": "Field quadrant of the target, or unspecified if not stated.",
                },
                "avoid_region": {
                    "type": "string",
                    "enum": ["NW", "NE", "SW", "SE", "N", "S", "E", "W", "none"],
                    # v3 replaced the single `constraints` enum with one scalar field per
                    # constraint. Reasons, in order of weight:
                    #   1. A judge can state both at once ("stay in the north half, keep clear
                    #      of the southwest corner") and a single-valued enum cannot hold it.
                    #      A list could, but a list enlarges the output space to a power set,
                    #      breaks the fixed-structure grammar that killed out-of-enum actions,
                    #      and makes exact-match scoring order-dependent. Separate scalars get
                    #      the expressiveness with none of that.
                    #   2. It removes a whole class of impossible output. `constraints:
                    #      avoid_regions` alongside an until_region was nonsense the grammar
                    #      could not forbid; here there is no cross-field contradiction to
                    #      train against.
                    #   3. Per-field accuracy becomes diagnostic -- avoid and contain fail
                    #      separately instead of hiding inside one `constraints` number.
                    # "none" means no keep-out constraint was stated. There is deliberately no
                    # value for "a region constraint with no region named": all 212
                    # avoid_regions rows through v11 named a compass region, so the case is
                    # unattested. If a judge ever says "avoid the shaded areas" with no
                    # compass reference, this enum needs a value for it.
                    "description": "Region the vehicle must keep out of, or none if no keep-out region was stated.",
                },
                "stay_region": {
                    "type": "string",
                    "enum": ["NW", "NE", "SW", "SE", "N", "S", "E", "W",
                             "pathway", "none"],
                    # Containment, the dual of avoid_region. The rules list it twice in their
                    # own example Leader's Intent statements ("Stay to the far south of the
                    # field until you reach the eastern half...", "...without touching other
                    # objects or leaving the safe pathway"), so it is judge-issued, which is
                    # the test for earning a slot.
                    #
                    # Halves (N/S/E/W) are here because the rules phrase constraints that way
                    # -- "the far south of the field", "the eastern half" -- while targets are
                    # only ever given by quadrant, which is why target_location stays
                    # NW/NE/SW/SE. "pathway" carries "the safe pathway": a corridor the rules
                    # name but never define geometrically, so it cannot be a compass value;
                    # the autonomy stack resolves what the pathway is.
                    #
                    # NOT modelled here: staying inside the overall ALLOWED AREA. The rules
                    # make that mandatory every run ("The system must stay within the allowed
                    # area"), so like object avoidance it is a standing invariant, and putting
                    # an always-true value in a field is exactly what made avoid_objects
                    # harmful -- 22 of 28 remaining field errors sat on that boundary.
                    "description": "Region the vehicle must remain inside, or none if no containment region was stated.",
                },
                "until_region": {
                    "type": "string",
                    "enum": ["NW", "NE", "SW", "SE", "N", "S", "E", "W", "none"],
                    # The release condition on a containment constraint: where the vehicle has
                    # to get to before stay_region stops applying ("stay to the far south UNTIL
                    # YOU REACH THE EASTERN HALF"). Only meaningful alongside stay_region, and
                    # "none" on every row without one -- including every avoid_region row,
                    # since a keep-out region has no release condition in any rules example.
                    # The grammar cannot enforce that dependency (GBNF has no cross-slot
                    # conditions), so training carries it.
                    #
                    # No "pathway" value: a pathway is somewhere you remain, not somewhere you
                    # arrive.
                    "description": "Region that releases the stay_region constraint once reached, or none if the constraint has no stated end.",
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
            "required": ["target_color", "avoid_region", "stay_region",
                         "until_region", "target_location", "action"],
        },
    },
}

# Serialization order. The three constraint fields sit together so a reader (and the grammar)
# sees the whole constraint in one run. Sentinels: the region fields use "none" (no such
# constraint); target_color and target_location keep "unspecified" (nothing stated), because
# "no target named" is not the same claim as "no constraint imposed".
FIELDS = ("target_color", "avoid_region", "stay_region", "until_region",
          "target_location", "action")
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
