"""Builds data/train_set.jsonl -- template x slot-value paraphrases, distinct
wording from data/eval_set.jsonl. Verifies zero text overlap with the eval set
before writing (train/eval separation is load-bearing for the eval numbers)."""
import json

COLORS = ["blue", "red", "yellow", "black"]
LOCS = ["NW", "NE", "SW", "SE"]
LOC_WORDS = {"NW": "northwest", "NE": "northeast", "SW": "southwest", "SE": "southeast"}
# The real target objects are foam dice (section 2.7), but the rules doc's own
# Leader's Intent phrasing calls them "boxes" (section 2.2), and a judge could
# just as plausibly say "cube" or "block" informally. Rotate through all of
# them so the model doesn't overfit to one noun for the target object.
NOUNS = ["die", "box", "cube", "block"]

rows = []

def add(text, color, constraints, loc, action):
    rows.append({"text": text, "expected": {
        "target_color": color, "constraints": constraints,
        "target_location": loc, "action": action,
    }})

# ---------- target retrieval (no location slot) ----------
no_loc_templates = [
    ("Locate the {c} target and carry it back to the starting zone, avoid touching anything else.", "avoid_objects"),
    ("{C} is the objective. Pick it up and return to base without disturbing the other objects.", "avoid_objects"),
    ("Go secure the {c} object and bring it home.", "none"),
    ("The {c} {n} is your target -- retrieve it and come back to the start.", "none"),
    ("All I need is the chip data off the {c} target, no need to bring the object back.", "none"),
    ("Scan the {c} object's NFC tag and transmit the code -- leave it in place.", "none"),
    ("Retrieve the {c} piece and get it to the starting square, that's the whole job.", "none"),
]
for template, constraint in no_loc_templates:
    action = "read_chip" if "chip" in template or "NFC" in template else "retrieve"
    for i, c in enumerate(COLORS):
        add(template.format(c=c, C=c.capitalize(), n=NOUNS[i % len(NOUNS)]), c, constraint, "unspecified", action)

loc_templates = [
    ("Fetch the {c} target sitting in the {lw} area and return to the starting zone.", "none"),
    ("Snag the {c} {n} near the {lw} corner, avoid the rest of the field, and bring it back.", "avoid_objects"),
    ("Swing by the {lw} corner, grab the {c} target, and head back without hitting anything.", "avoid_objects"),
]
for i, (template, constraint) in enumerate(loc_templates):
    for j, c in enumerate(COLORS):
        loc = LOCS[(i + j) % len(LOCS)]
        noun = NOUNS[(i + j) % len(NOUNS)]
        add(template.format(c=c, lw=LOC_WORDS[loc], n=noun), c, constraint, loc, "retrieve")

# ---------- abort ----------
abort_templates = [
    "Abort now, hold your position and don't move.",
    "This run is scrubbed -- power down and stay put.",
    "Cut the mission short, remain hidden where you are.",
    "We're done here, stop and stay concealed.",
    "Emergency stop -- abort and remain undetected.",
    "Mission's canceled, freeze and wait for further instructions.",
    "Break off the approach, hold position quietly.",
    "Shut everything down right where you are.",
    "Cease operation immediately and hold position.",
    "That's a wrap on this attempt -- abort and lay low.",
    "Cancel the run right where you are, don't advance.",
    "We need eyes off you -- abort and go still.",
    "Stop everything and remain out of sight.",
    "This mission is finished, power down in place.",
]
for t in abort_templates:
    add(t, "unspecified", "none", "unspecified", "abort")

return_to_start_templates = [
    "Drop the objective and make your way back to the starting zone.",
    "Recall the vehicle, bring it back to base now.",
    "That's enough for today, return the system to start.",
    "Pull the system out and send it back to the starting square.",
    "Bring the system home, we're aborting the retrieval.",
    "Head back to the starting zone, we're calling off the search.",
]
for t in return_to_start_templates:
    add(t, "unspecified", "none", "unspecified", "return_to_start")

# ---------- avoid region ----------
avoid_region_templates = [
    "Keep the vehicle out of the {lw} corner from here on.",
    "The {lw} area is restricted, plan your route around it.",
    "Judges have flagged the {lw} section -- avoid it.",
    "Don't let the system anywhere near the {lw} quadrant.",
    "New boundary: nothing enters the {lw} region.",
    "Treat the {lw} corner as a keep-out zone.",
    "Reroute around the {lw} side of the field, it's off limits now.",
    "The {lw} quarter just became a restricted zone.",
    "Avoid crossing into the {lw} section under any circumstance.",
    "That {lw} area is unsafe, route the vehicle elsewhere.",
]
for template in avoid_region_templates:
    for loc in LOCS:
        add(template.format(lw=LOC_WORDS[loc]), "unspecified", "avoid_regions", loc, "retrieve")

# ---------- object relocated ----------
relocated_templates = [
    "Field update: the {c} target now sits in the {lw} corner.",
    "The {c} object was moved -- it's currently in the {lw} section.",
    "Correction on target position: {c} target, {lw} quadrant.",
    "The {c} {n} relocated to the {lw} area, adjust your route.",
    "Latest report has the {c} target in the {lw} corner now.",
    "Position change for the {c} object: {lw} quarter of the field.",
    "The {c} target isn't where it was, it's now near the {lw} corner.",
    "Update your plan -- {c} target moved to the {lw} side.",
    "New coordinates for the {c} object: {lw} region.",
    "Just relocated: the {c} target is now in the {lw} corner.",
]
for i, template in enumerate(relocated_templates):
    for j, c in enumerate(COLORS):
        loc = LOCS[(i + j) % len(LOCS)]
        noun = NOUNS[(i + j) % len(NOUNS)]
        add(template.format(c=c, lw=LOC_WORDS[loc], n=noun), c, "none", loc, "retrieve")

# ---------- rebalance: read_chip / return_to_start / abort were badly under-represented
# (112 retrieve vs 8 read_chip vs 6 return_to_start), so the model had almost no signal
# for half the action vocabulary. Written by category from the rules doc, not by
# inspecting eval failures. ----------
read_chip_no_loc = [
    "Get a chip reading from the {c} {n} and leave it in place.",
    "We need the tag ID off the {c} target, don't move it.",
    "Scan the {c} object's chip and report the number.",
    "Read the tag on the {c} target -- no retrieval this run.",
    "Tag data only from the {c} {n}, leave the object alone.",
    "Record the chip on the {c} target and stand by.",
]
for i, template in enumerate(read_chip_no_loc):
    for j, c in enumerate(COLORS):
        add(template.format(c=c, n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", "unspecified", "read_chip")

read_chip_loc = [
    "Scan the chip on the {c} target in the {lw} corner and send it up.",
    "Take a tag reading from the {c} {n} in the {lw} quadrant.",
    "Chip data only from the {c} object in the {lw} section.",
    "Pull the code off the {c} target over in the {lw} area.",
]
for i, template in enumerate(read_chip_loc):
    for j, c in enumerate(COLORS):
        loc = LOCS[(i + j) % len(LOCS)]
        add(template.format(c=c, lw=LOC_WORDS[loc], n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", loc, "read_chip")

more_return_to_start = [
    "Bring the vehicle back to the starting zone, leave the target behind.",
    "Return to the launch point now.",
    "Come back to start, we're resetting the run.",
    "Send it home -- the objective can stay where it is.",
    "Back to the starting square, no retrieval needed.",
    "Abandon the pickup and return to base.",
    "Drive the system back to the start line.",
    "Get the vehicle home, we'll re-run this later.",
    "Recall the rover to the starting zone.",
    "Head back to the launch box and hold there.",
    "Leave the object and come home.",
    "Take the system back to start, empty is fine.",
    "Come back to the starting area, we're done attempting this.",
    "Reset -- bring the vehicle back to where it started.",
    "Withdraw to the starting zone without the target.",
    "Return the vehicle to base, we're not collecting anything.",
    "Bring it in. No pickup this run.",
    "Come back to the start point and wait there.",
    "Walk it back to the starting zone, we're done searching.",
    "Send the system home and leave the field as it is.",
    "Return to the start, we're switching to the next attempt.",
    "Get back to base, the retrieval is off.",
    "Drive home -- forget what you were after.",
    "Pull the vehicle back to the launch zone now.",
]
for t in more_return_to_start:
    add(t, "unspecified", "none", "unspecified", "return_to_start")

more_abort = [
    "Hard abort, stop where you are.",
    "No further movement -- the run is over.",
    "Stop the vehicle and keep it dark.",
    "We've lost clearance, shut it down in place.",
    "End of run. Hold still and stay hidden.",
    "Terminate now, no movement, no signal.",
    "Knock it off and hold position.",
    "Remain stationary and concealed, the attempt is aborted.",
    "That's enough, kill the run and stay low.",
    "Discontinue the attempt, freeze where you are.",
    "Take it offline right now, don't move.",
    "Halt. Nothing moves until further notice.",
    "Cut power and stay out of view.",
    "Stop immediately, stay dark.",
    "Call off everything and remain in place.",
    "Abort the approach and keep the system hidden.",
]
for t in more_abort:
    add(t, "unspecified", "none", "unspecified", "abort")

# compound "do X and then come back" phrasing still resolves to retrieve --
# the return leg is part of the retrieval, not a separate return_to_start
compound_retrieve = [
    "Collect the {c} {n} and then head straight back to the start.",
    "Pick up the {c} target, then return to the starting zone.",
]
for i, template in enumerate(compound_retrieve):
    for j, c in enumerate(COLORS):
        add(template.format(c=c, n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", "unspecified", "retrieve")


def main():
    with open("data/eval_set.jsonl", encoding="utf-8") as f:
        eval_texts = {json.loads(line)["text"] for line in f}

    overlap = [r["text"] for r in rows if r["text"] in eval_texts]
    if overlap:
        raise SystemExit(f"train/eval overlap detected, aborting write: {overlap}")

    with open("data/train_set.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} examples to data/train_set.jsonl (0 overlap with eval_set.jsonl)")

if __name__ == "__main__":
    main()
