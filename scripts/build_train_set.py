"""Builds data/train_set.jsonl -- template x slot-value paraphrases, distinct
wording from data/eval_set.jsonl. Verifies zero text overlap with the eval set
before writing (train/eval separation is load-bearing for the eval numbers)."""
import json

from schema import expand_v2

COLORS = ["blue", "red", "yellow", "black"]
LOCS = ["NW", "NE", "SW", "SE"]
LOC_WORDS = {"NW": "northwest", "NE": "northeast", "SW": "southwest", "SE": "southeast"}
# The real target objects are foam dice (section 2.7), but the rules doc's own
# Leader's Intent phrasing calls them "boxes" (section 2.2), and a judge could
# just as plausibly say "cube" or "block" informally. Rotate through all of
# them so the model doesn't overfit to one noun for the target object.
NOUNS = ["die", "box", "cube", "block"]

rows = []

# Object avoidance (rules doc 2.2) and update_only (round 8) both left the schema, and
# schema v3 split `constraints` into avoid_region/stay_region/until_region. The label tuples
# below are still written in the ORIGINAL v2 shape -- expand_v2() in schema.py does all three
# mappings at write time, so the labelling intent stays readable and every mapping is in one
# reviewable place instead of smeared over ~40 call sites.
def add(text, color, constraints, loc, action, stay="none", until="none", avoid="none"):
    rows.append({"text": text,
                 "expected": expand_v2(color, constraints, loc, action, stay, until, avoid)})


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
    action = "read_chip" if "chip" in template or "NFC" in template else "collect_target"
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
        add(template.format(c=c, lw=LOC_WORDS[loc], n=noun), c, constraint, loc, "collect_target")

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
        add(template.format(lw=LOC_WORDS[loc]), "unspecified", "avoid_regions", loc, "collect_target")

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
        # a relocation notice updates target_location on an active task -> update_only
        add(template.format(c=c, lw=LOC_WORDS[loc], n=noun), c, "none", loc, "update_only")

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
            c, "none", "unspecified", "collect_target")


# ---------- round 2 rebalance. avoid_objects was 16 examples against 172 "none", and every
# one of them used literal "avoid" / "without touching" wording, so the model keyed on the
# verb rather than the meaning. These state the same constraint in other words. ----------
avoid_objects_no_loc = [
    "Bring back the {c} {n} and make sure you don't bump any of the other objects.",
    "Collect the {c} target, nothing else on the field may be contacted.",
    "Get the {c} target home without making contact with any other piece.",
    "The {c} object is yours -- leave every other item untouched.",
    "Pick up the {c} {n}, steer around the rest of the objects.",
    "Recover the {c} target and stay clear of the other dice.",
    "Fetch the {c} object; the other pieces are off limits.",
    "Retrieve the {c} {n} and keep the remaining objects undisturbed.",
]
for i, template in enumerate(avoid_objects_no_loc):
    for j, c in enumerate(COLORS):
        add(template.format(c=c, n=NOUNS[(i + j) % len(NOUNS)]),
            c, "avoid_objects", "unspecified", "collect_target")

avoid_objects_loc = [
    "Grab the {c} target in the {lw} corner and keep off the other objects.",
    "Collect the {c} {n} from the {lw} section without disturbing anything else.",
]
for i, template in enumerate(avoid_objects_loc):
    for j, c in enumerate(COLORS):
        loc = LOCS[(i + j) % len(LOCS)]
        add(template.format(c=c, lw=LOC_WORDS[loc], n=NOUNS[(i + j) % len(NOUNS)]),
            c, "avoid_objects", loc, "collect_target")

# ---------- the retrieve / return_to_start boundary. The rule is whether a target object is
# NAMED: "take the black cube back to the start" is a retrieval whose return leg is mentioned,
# not a bare recall. Round 1 added 30 return_to_start examples against only 8 compound
# retrieves, which pushed the model to read "back to the start" as return_to_start. ----------
object_named_returns = [
    "Escort the {c} {n} back to the starting zone.",
    "The {c} object comes back with the vehicle.",
    "Move the {c} target to the start point and leave it there.",
    "Transport the {c} {n} to the starting square.",
    "Walk the {c} object back to base.",
    "Ferry the {c} target to the start line.",
    "The {c} {n} goes back to the starting zone with you.",
    "Bring the {c} piece in to the start.",
]
for i, template in enumerate(object_named_returns):
    for j, c in enumerate(COLORS):
        add(template.format(c=c, n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", "unspecified", "collect_target")


# ---------- round 3. Every avoid_objects example above states a PROHIBITION ("avoid",
# "don't bump", "off limits"), so the model keyed on negative-polarity verbs and missed the
# constraint when it was phrased as a permission granted to the target instead. It also never
# saw avoid_objects alongside spatial words, so "near the other dice" pulled it to
# avoid_regions. Written by category, with wording deliberately distinct from the eval
# sentences that exposed the gap -- these must measure generalisation, not recall. ----------

# the target is named as the ONE permitted object rather than the others being forbidden
exclusive_permission = [
    "The {c} {n} is the only object you're cleared to handle -- bring it in.",
    "Only the {c} target may be moved; everything else stays exactly as it is.",
    "You have permission to make contact with the {c} object and nothing more.",
    "The {c} {n} alone is fair game -- return it to the starting zone.",
]
for i, template in enumerate(exclusive_permission):
    for j, c in enumerate(COLORS):
        add(template.format(c=c, n=NOUNS[(i + j) % len(NOUNS)]),
            c, "avoid_objects", "unspecified", "collect_target")

# route verbs carrying the constraint without any "avoid"-family cue word
routed_around_objects = [
    "Work your way around the other objects and bring the {c} {n} home.",
    "Bring the {c} object back, routing wide of the other pieces out there.",
]
for i, template in enumerate(routed_around_objects):
    for j, c in enumerate(COLORS):
        add(template.format(c=c, n=NOUNS[(i + j) % len(NOUNS)]),
            c, "avoid_objects", "unspecified", "collect_target")

# avoid_objects stated with proximity/route language -- the disambiguator is that the thing
# being kept away from is the OTHER OBJECTS, not a named region of the field
proximity_to_objects = [
    "No close passes near the other pieces -- collect the {c} target and return.",
    "Keep your distance from the other objects while you fetch the {c} {n}.",
]
for i, template in enumerate(proximity_to_objects):
    for j, c in enumerate(COLORS):
        add(template.format(c=c, n=NOUNS[(i + j) % len(NOUNS)]),
            c, "avoid_objects", "unspecified", "collect_target")

# avoid_objects also occurs on read_chip runs, which train had no example of
read_chip_avoid_objects = [
    "Chip scan on the {c} {n} only -- the other pieces stay untouched.",
]
for i, template in enumerate(read_chip_avoid_objects):
    for j, c in enumerate(COLORS):
        add(template.format(c=c, n=NOUNS[(i + j) % len(NOUNS)]),
            c, "avoid_objects", "unspecified", "read_chip")

# hard negatives from the region side: a restricted region that mentions objects, so
# "object" vocabulary alone does not decide avoid_objects
region_mentioning_objects = [
    "The {lw} corner is closed off, even if an object is sitting in there.",
    "Route around the {lw} section; whatever is inside it is out of play.",
]
for template in region_mentioning_objects:
    for loc in LOCS:
        add(template.format(lw=LOC_WORDS[loc]), "unspecified", "avoid_regions", loc, "collect_target")

# ---------- directional distractors. Every location example above contains exactly one
# direction word, so the model learned "first direction mentioned = target_location" and a
# judge describing an approach route before naming the target flipped the slot. The route
# direction is a plain cardinal, the target sits in a named corner -- the cue is which one
# the object is attached to. PATH_WORD never shares a compass component with its target
# corner, so the two are never genuinely ambiguous. ----------
PATH_WORD = {"NW": "southern", "NE": "western", "SW": "eastern", "SE": "northern"}
distractor_route = [
    "Hug the {pw} edge on the way out, then collect the {c} {n} from the {lw} corner.",
    "Travel along the {pw} side of the field and pick up the {c} target in the {lw} quadrant.",
    "Approach from the {pw} end -- the {c} {n} is in the {lw} corner.",
    "Cross the {pw} half first, then retrieve the {c} object sitting in the {lw} section.",
]
for i, template in enumerate(distractor_route):
    for j, c in enumerate(COLORS):
        loc = LOCS[(i + j) % len(LOCS)]
        add(template.format(c=c, pw=PATH_WORD[loc], lw=LOC_WORDS[loc],
                            n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", loc, "collect_target")

# hard negatives for the round-3 additions: proximity and route words ("near", "around",
# "past") in sentences that carry NO constraint at all, so the new avoid_objects templates
# above do not teach the model that this vocabulary implies a constraint by itself
generic_obstacle_none = [
    "Collect the {c} {n} and navigate past any obstacles on the way back.",
    "Get the {c} target and work through whatever is in your path.",
]
for i, template in enumerate(generic_obstacle_none):
    for j, c in enumerate(COLORS):
        add(template.format(c=c, n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", "unspecified", "collect_target")

unconstrained_proximity = [
    "The {c} {n} is somewhere near the {lw} corner -- bring it back.",
    "Pick up the {c} target over by the {lw} corner.",
    "You'll go past the {lw} section; the {c} {n} is waiting there.",
    "Swing out to the {lw} quadrant and collect the {c} object.",
]
for i, template in enumerate(unconstrained_proximity):
    for j, c in enumerate(COLORS):
        loc = LOCS[(i + j) % len(LOCS)]
        add(template.format(c=c, lw=LOC_WORDS[loc], n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", loc, "collect_target")


# ---------- round 4. Two findings from data/heldout_set.jsonl, the first eval set that was
# never used for model selection.
#
# (a) Held-out recall tracked training share almost linearly -- none 63.5% of train -> 0.970
# recall, avoid_objects 24.0% -> 0.783, avoid_regions 12.5% -> 0.762; retrieve 70.8% -> 0.993,
# abort 7.8% -> 0.842. The model had learned the label prior, not just the task, and 15 of its
# 19 constraint errors collapsed to the majority class. Hence the balancing step in main().
#
# (b) avoid_regions still had the exact bug round 3 fixed for avoid_objects: every region
# template above carries an explicit prohibition verb (avoid / keep out / restricted / don't),
# so stative and indirect region statements were read as no constraint at all. Same fix,
# applied to the other class. ----------

# region restrictions with no prohibition verb: stative, passive, hedged, or bureaucratic
regions_without_avoid_verb = [
    "The {lw} quarter is shut for this attempt.",
    "{LW} is no longer part of the course.",
    "Officials have fenced off the {lw} section.",
    "{LW} corner: no entry.",
    "The {lw} region is unavailable for the rest of the round.",
    "We have lost access to the {lw} quadrant.",
    "The {lw} side is sealed until further notice.",
    "Consider the {lw} corner gone from the map.",
    "{LW} quadrant is inactive this round.",
    "The {lw} area has been taken out of service.",
    "It would be better if the {lw} corner went untouched.",
    "We would prefer the vehicle stayed clear of the {lw} area.",
    "No part of your route may include the {lw} section.",
    "The {lw} region is to be treated as sealed.",
    "Scratch the {lw} quarter from your plan.",
    "The {lw} corner has been pulled from play.",
]
for template in regions_without_avoid_verb:
    for loc in LOCS:
        add(template.format(lw=LOC_WORDS[loc], LW=LOC_WORDS[loc].capitalize()),
            "unspecified", "avoid_regions", loc, "collect_target")

# abort and return_to_start were the two weakest actions on held-out (0.842 / 0.950) and the
# two smallest classes (7.8% each). More wording, not just more copies.
round4_abort = [
    "Belay that, hold where you are.",
    "Scrub it. No movement.",
    "We are off -- stop and stay covered.",
    "Stand by in place, the attempt is finished.",
    "Nothing further this run. Hold and stay quiet.",
    "Cut it there and keep out of sight.",
    "The window is gone -- stop the system.",
    "Drop everything, stay exactly where you are.",
    "Quit the run and keep the vehicle dark.",
    "All stop. Remain concealed.",
    "We are done attempting this. Freeze in place.",
    "Bring it to a halt and stay unseen.",
    "No more of this run -- stop moving.",
    "Stay put, the attempt is over.",
    "Shut down and hold, we have been flagged.",
    "That is the end of it. Stop and stay hidden.",
    "Do not continue. Hold position.",
    "Cancel out and keep still.",
    "Stop right there and stay dark.",
    "We are pulling the run. Hold and conceal.",
]
for t in round4_abort:
    add(t, "unspecified", "none", "unspecified", "abort")

round4_return_to_start = [
    "Just bring the vehicle in, nothing else.",
    "Back to the launch area, empty-handed.",
    "Come in. There is nothing to collect.",
    "Walk it home, we are not picking anything up.",
    "Return the system to the start and hold.",
    "Get it back to the launch square.",
    "Bring the rover in, the search is off.",
    "Come home now, leave everything out there.",
    "Take it back to the start, no cargo.",
    "Withdraw to base, we are resetting.",
    "Head in. We will try again later.",
    "Pull it back to the start zone and park.",
    "Return to launch, nothing to carry.",
    "Bring the system back, we are calling it.",
    "Come back to the start box and wait there.",
    "Drive it in, the collection is cancelled.",
    "Home you go, nothing to bring back.",
    "Get the vehicle to the start and stop.",
    "Back to base, we are not retrieving anything.",
    "Return empty to the starting area.",
]
for t in round4_return_to_start:
    add(t, "unspecified", "none", "unspecified", "return_to_start")


# (read_chip, avoid_objects) had only 4 distinct sentences, too few a pool to reweight
# without just memorising them four times over. More wording for the cell.
round4_read_chip_avoid_objects = [
    "Tag the {c} {n} and keep well off the other pieces.",
    "Chip reading on the {c} target, nothing else gets touched.",
    "Scan the {c} object only -- the rest stay exactly as they are.",
    "Pull the code from the {c} {n} without contacting the others.",
]
for i, template in enumerate(round4_read_chip_avoid_objects):
    for j, c in enumerate(COLORS):
        add(template.format(c=c, n=NOUNS[(i + j) % len(NOUNS)]),
            c, "avoid_objects", "unspecified", "read_chip")


# Held-out recall tracked training share almost linearly, so the prior itself is a bug worth
# fixing. Exactly even thirds are NOT reachable: abort and return_to_start always carry
# constraints="none", so if those two actions hold any meaningful share, "none" is floored
# well above a third. This is the closest reachable mix, not an even one.
# ponytail: oversample the minority classes rather than discarding majority rows -- throwing
# away real sentences to hit a ratio would cost coverage we already paid for.
# ---------- round 6. New mission-control actions. These were added because the Mission State
# Machine already describes behaviour the schema could not express (update_only), and because
# several operator commands had nowhere to land (status queries, pause/resume, retry).
#
# The design constraint, learned from avoid_objects: what hurts accuracy is not the NUMBER of
# classes but OVERLAPPING SURFACE FORMS between them. constraints had 3 values and sat at 0.78
# because two of them shared wording; action had 4 values and sat at 0.99 because none did.
# So every template below is written to stay lexically clear of its nearest neighbour, and the
# two genuinely competing pairs (pause/abort, update_only/collect_target) get explicit
# contrast examples rather than being left to chance. ----------

# update_only: a STATEMENT that corrects location or constraints on a task already running.
# The relocation templates above already carry most of this class; these are the explicit
# "just update, don't restart" phrasings.
update_only_templates = [
    "Adjust the target location to the {lw} corner, keep going.",
    "Correction only: the {c} {n} is in the {lw} quadrant now.",
    "Same task, new position -- {c} target, {lw} section.",
    "Update the plan to the {lw} area, don't restart the run.",
    "Revised location for the {c} object: {lw} corner. Continue as you were.",
    "Just a position update -- {c} {n}, {lw} side.",
]
for i, template in enumerate(update_only_templates):
    for j, c in enumerate(COLORS):
        loc = LOCS[(i + j) % len(LOCS)]
        add(template.format(c=c, lw=LOC_WORDS[loc], n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", loc, "update_only")

# report_status: a query, not an order. Shares vocabulary with nothing else in the schema.
report_status_templates = [
    "Report your status.",
    "What's your current status?",
    "Give me a status update.",
    "Where are you and what are you doing?",
    "Check in -- what's the current intent?",
    "Status report, please.",
    "Tell me what you're working on right now.",
    "What's the mission state?",
    "Read back your current orders.",
    "Confirm what you think the task is.",
    "How's it going out there?",
    "Sitrep.",
    "What have you got so far?",
    "Talk to me -- where do things stand?",
    "Say again your current objective.",
    "I need to know what you're doing.",
    "Give me a readback of the current task.",
    "What's the plan right now?",
    "Where's the vehicle and what's it after?",
    "Update me on where things stand.",
    "Current objective?",
    "What are you tracking at the moment?",
    "Let me know what's happening out there.",
    "Brief me on the current task.",
]
for t in report_status_templates:
    add(t, "unspecified", "none", "unspecified", "report_status")

# pause: TEMPORARY, resumption expected. The competing class is abort, which ends the run --
# so these deliberately carry waiting/temporary language and never concealment or finality.
pause_templates = [
    "Hold on a second.",
    "Wait there for a moment.",
    "Pause the run, I'll come back to you.",
    "Stand by.",
    "Give me a minute -- hold where you are.",
    "Freeze for now, more to follow.",
    "Hold one.",
    "Take a break, we're not done.",
    "Suspend the task for a moment.",
    "Wait one, I need to check something.",
    "Hold that thought and stay put.",
    "Temporarily stop, I'll tell you when to go.",
    "Just wait, don't do anything yet.",
    "Stop for now, stand by for instructions.",
    "Hold the task, more coming.",
    "Pause right there.",
    "Hold up a moment.",
    "Park it for now, I'll be back.",
    "Sit tight for a second.",
]
for t in pause_templates:
    add(t, "unspecified", "none", "unspecified", "pause")

resume_templates = [
    "Carry on.",
    "Resume the task.",
    "Go ahead, continue.",
    "Pick up where you left off.",
    "You're clear to continue.",
    "Back to it.",
    "Continue the run.",
    "As you were -- keep going.",
    "Proceed.",
    "Start again from where you stopped.",
    "Resume, same objective.",
    "Green light, carry on.",
    "Keep going with what you had.",
    "Continue as before.",
    "Go on.",
    "Unpause and continue.",
    "Off you go again.",
    "Restart the task from where it paused.",
    "You're good to go.",
    "Resume operations.",
    "Get moving again.",
    "Back to work.",
    "Carry on with the objective.",
    "Continue on, same as before.",
]
for t in resume_templates:
    add(t, "unspecified", "none", "unspecified", "resume")

# retry_read / retry_send: explicit repetition of one step. The cue is "again"/"re-" plus
# which step, so these stay clear of read_chip (a first read) and of each other.
retry_read_templates = [
    "Try that scan again.",
    "Read the chip one more time.",
    "That tag read failed -- do it again.",
    "Re-scan the chip.",
    "Take another reading off the tag.",
    "Didn't get that -- read the chip again.",
    "Repeat the chip read.",
    "Scan it again, the first one didn't take.",
    "Another tag read, please.",
    "Retry the NFC read.",
    "Do the scan over.",
    "One more attempt on the chip.",
    "Give the chip another go.",
    "Read that tag once more.",
    "Second attempt on the scan.",
    "Run the chip read again.",
    "Have another go at the tag.",
    "Scan once more, please.",
]
for t in retry_read_templates:
    add(t, "unspecified", "none", "unspecified", "retry_read")

retry_send_templates = [
    "Send that code again.",
    "Resend the message.",
    "That transmission didn't arrive -- try again.",
    "Re-transmit the tag data.",
    "Send it one more time.",
    "We didn't receive it, send again.",
    "Retry the upload.",
    "Push that code through again.",
    "Transmit again, please.",
    "The send failed -- repeat it.",
    "Try sending that once more.",
    "Re-send the chip data.",
    "That upload didn't land -- again.",
    "Put the message through a second time.",
    "Send the reading once more, please.",
    "Transmit it over again.",
    "Fire that message off again.",
    "Send the tag data once more.",
    "Have another go at transmitting.",
    "Put the code through again.",
    "Repeat the transmission.",
    "Try the send again.",
]
for t in retry_send_templates:
    add(t, "unspecified", "none", "unspecified", "retry_send")

# explicit contrast pairs for the two boundaries most likely to blur
pause_vs_abort_contrast = [
    ("Stop and hold, I'll be right back.", "pause"),
    ("Stop everything and stay out of sight.", "abort"),
    ("Hold position, more instructions coming.", "pause"),
    ("Hold position and stay hidden, the run is over.", "abort"),
    ("Wait there, we're not finished.", "pause"),
    ("We're finished -- power down where you are.", "abort"),
    ("Freeze, I need a moment.", "pause"),
    ("Freeze and stay concealed, mission scrubbed.", "abort"),
]
for t, a in pause_vs_abort_contrast:
    add(t, "unspecified", "none", "unspecified", a)

update_vs_collect_contrast = [
    ("The {c} {n} is now in the {lw} corner.", "update_only"),
    ("Go get the {c} {n} from the {lw} corner.", "collect_target"),
    ("{C} target has shifted to the {lw} section.", "update_only"),
    ("Bring me the {c} target from the {lw} section.", "collect_target"),
]
for i, (template, a) in enumerate(update_vs_collect_contrast):
    for j, c in enumerate(COLORS):
        loc = LOCS[(i + j) % len(LOCS)]
        add(template.format(c=c, C=c.capitalize(), lw=LOC_WORDS[loc],
                            n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", loc, a)


# ---------- round 7. Three boundaries failed in round 6, and two of them are the SAME BUG
# this project has now hit three times: a class that depends on one cue word rather than on
# meaning. avoid_objects keyed on prohibition verbs, avoid_regions keyed on "avoid", and now
# update_only keys on an explicit update marker. The fix each time is wording that carries
# the meaning without the cue -- plus, where two classes genuinely compete, matched pairs
# that differ only in the deciding word. ----------

# (a) update_only beyond the ten "Field update:/Correction on:/relocated" templates above.
# Per schema.py, update_only REQUIRES a marker -- a bare locative is collect_target -- so
# these vary the marker instead of dropping it.
update_marker_templates = [
    "Be advised, the {c} {n} is in the {lw} corner now.",
    "Heads up -- {c} target has shifted to the {lw} quadrant.",
    "Amendment: {c} object, {lw} section.",
    "Scratch that, the {c} {n} is in the {lw} area.",
    "Actually it's the {lw} corner for the {c} target.",
    "Revised: {c} {n} in the {lw} quadrant.",
    "Change of position -- the {c} object is {lw} now.",
    "Disregard the old location; the {c} target is {lw}.",
    "One correction: the {c} {n} sits in the {lw} corner.",
    "FYI the {c} object ended up in the {lw} section.",
    "Belay the last position, {c} target is in the {lw} corner.",
    "Note a change: {c} {n}, {lw} side.",
]
for i, template in enumerate(update_marker_templates):
    for j, c in enumerate(COLORS):
        loc = LOCS[(i + j) % len(LOCS)]
        add(template.format(c=c, C=c.capitalize(), lw=LOC_WORDS[loc],
                            n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", loc, "update_only")

# the other half of the rule: a BARE locative is a tasking, not an update. These exist
# already in the retrieve sections, but not enough of them to hold the line against the
# marker templates above.
bare_locative_is_collect = [
    "The {c} {n} is in the {lw} corner.",
    "{C} target sits in the {lw} quadrant.",
    "The {c} object is over in the {lw} section.",
    "{C} {n} is up in the {lw} area.",
]
for i, template in enumerate(bare_locative_is_collect):
    for j, c in enumerate(COLORS):
        loc = LOCS[(i + j) % len(LOCS)]
        add(template.format(c=c, C=c.capitalize(), lw=LOC_WORDS[loc],
                            n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", loc, "collect_target")

# (b) read_chip stated by EXCLUSION rather than by a scan verb. "chip only" and
# "tag, not the object" were the last failures standing after round 5 and they survived
# round 6 -- the meaning is carried by what is ruled out, which is the same shape that made
# avoid_objects hard.
read_chip_elliptical = [
    "Tag only on the {c} {n}.",
    "Chip data alone from the {c} target.",
    "Just the code off the {c} object, nothing else.",
    "{C} {n}: chip, not the object.",
    "Nothing but the tag from the {c} target.",
    "Only the chip on the {c} {n}, leave it where it is.",
    "The {c} object's code is all we want.",
    "Purely a tag read on the {c} target.",
    "{C} target -- read, don't collect.",
    "Code from the {c} {n} and that is the whole job.",
]
for i, template in enumerate(read_chip_elliptical):
    for j, c in enumerate(COLORS):
        add(template.format(c=c, C=c.capitalize(), n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", "unspecified", "read_chip")

# the mirror: "only" also appears on collections, so it must not become a read_chip cue
only_is_collect = [
    "Only the {c} {n} comes back with you.",
    "The {c} target is the only thing to collect.",
    "Bring just the {c} object home.",
    "Nothing but the {c} {n} needs collecting.",
]
for i, template in enumerate(only_is_collect):
    for j, c in enumerate(COLORS):
        add(template.format(c=c, n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", "unspecified", "collect_target")

# (c) report_status collided with read_chip on the verb "report" -- round 6 read
# "Report the tag ID from the blue cube" as a status query. The deciding cue is the OBJECT
# of the verb: a tag/code/chip is read_chip, the vehicle's own state is report_status.
report_verb_contrast = [
    ("Report the tag number from the {c} {n}.", "read_chip"),
    ("Report back the {c} target's chip code.", "read_chip"),
    ("Send me the {c} object's tag reading.", "read_chip"),
    ("Give me the code off the {c} {n}.", "read_chip"),
]
for i, (template, a) in enumerate(report_verb_contrast):
    for j, c in enumerate(COLORS):
        add(template.format(c=c, n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", "unspecified", a)

report_state_contrast = [
    "Report on what you're doing.",
    "Report your position and task.",
    "Send me a status update, not a tag.",
    "Give me your own status.",
    "Report back on the mission, not the chip.",
    "Tell me your state, not a code.",
]
for t in report_state_contrast:
    add(t, "unspecified", "none", "unspecified", "report_status")

# (d) pause failed whenever the sentence LED with stop-language, which abort owns. These
# put the temporary marker after the stop verb, which is the order that broke it.
pause_after_stop_verb = [
    "Stop for a moment, I'll be back.",
    "Halt the task briefly.",
    "Interrupt what you're doing, back shortly.",
    "Cease for now, we're not finished.",
    "Break off for a second, more coming.",
    "Everything stops briefly.",
    "Suspend for a moment.",
    "Cut the task for now, stand by.",
    "Freeze it for a minute, I'll call you back.",
    "Shut it down for a moment only.",
    "Stop work temporarily.",
    "Down tools for a second, not done yet.",
]
for t in pause_after_stop_verb:
    add(t, "unspecified", "none", "unspecified", "pause")

# the abort half of the same openings, so the temporary marker is the only difference
abort_after_stop_verb = [
    "Stop for good, we're done here.",
    "Halt the task, the run is over.",
    "Interrupt what you're doing and stay hidden.",
    "Cease now, nothing more today.",
    "Break off and keep out of sight.",
    "Everything stops, permanently.",
    "Suspend the run and conceal yourself.",
    "Cut the task, we've lost clearance.",
]
for t in abort_after_stop_verb:
    add(t, "unspecified", "none", "unspecified", "abort")


# Balancing one marginal skews the other: oversampling avoid_regions (always action
# "collect_target") pushed retrieve from 71% to 78% and squeezed read_chip down to 7.9%. So the
# targets are on the JOINT (action, constraints) cell instead.
#
# Exactly even is not reachable. abort and return_to_start only ever carry
# constraints="none", so any meaningful share for those two actions floors "none" well
# above a third. These targets are the best compromise across both fields: they take
# constraints from 63/24/13 to roughly 47/28/25, and actions from 71/14/8/8 to roughly
# 57/17/13/13, without letting either axis collapse.
# PREDICTION TO CHECK IN THE v13 RESULTS: target_location is now named on 14.7% of rows,
# down from 32.6% in v11. Nothing was deleted -- un-overloading moved the avoided region off
# target_location on 212 rows, and those rows had been the single largest source of "a compass
# word is in this sentence, so set target_location". The labels are right now and were wrong
# before, but the field has half the positive density it trained on at 97.70%. If
# target_location is the field that regresses in v13, that is the cause, and the fix is more
# genuine target-location sentences, NOT putting regions back in the slot.

# ---------- round 10: containment (stay_region, until_region) ----------
# schema v3 added these two fields with ZERO training examples. The only attested sentences
# are the rules doc's own two, and BOTH sit in data/eval_set.jsonl, so neither can be trained
# on. Everything below is written from the rules' phrasing PATTERN, not its sentences:
#   "Stay to the far south of the field until you reach the eastern half and then obtain
#    the blue object in the northeast corner."   -> stay=S, until=E, blue, NE
#   "...bring back the orange box without touching other objects or leaving the safe
#    pathway."                                   -> stay=pathway
#
# HALVES lead this block. The rules state containment in halves ("the far south", "the
# eastern half") while they only ever give targets by quadrant -- that asymmetry is why
# N/S/E/W exist on the constraint fields and not on target_location. Quadrant containment is
# here too, because a judge can obviously say "stay in the northwest corner", just less of it.
HALF_WORDS = {"N": "northern", "S": "southern", "E": "eastern", "W": "western"}
HALF_BARE = {"N": "north", "S": "south", "E": "east", "W": "west"}
HALVES = ["N", "S", "E", "W"]

# (a) containment with no release condition, stated in halves
stay_half_templates = [
    "Keep the vehicle in the {hw} half of the field for this run.",
    "Stay within the {hw} half the entire way out and back.",
    "Operate only in the {hw} half, nowhere else.",
    "The vehicle is confined to the {hw} side of the field.",
    "Hold to the far {hb} of the field throughout.",
    "Remain inside the {hw} half until I say otherwise.",
    "Your working area this round is the {hw} half, full stop.",
    "Do not leave the {hw} half of the course.",
]
for i, template in enumerate(stay_half_templates):
    for j, h in enumerate(HALVES):
        add(template.format(hw=HALF_WORDS[h], hb=HALF_BARE[h]),
            "unspecified", "none", "unspecified", "collect_target", stay=h)

# (b) containment stated in quadrants, and WITH a named target -- the co-occurrence that
# avoid_region never got in v11 (0 of its 212 rows named a colour), which is the leading
# suspect for why it failed every novel-phrasing sentence on heldout5 that named one.
stay_quad_templates = [
    "Grab the {c} {n} and stay inside the {lw} quadrant while you do it.",
    "The {c} target is yours -- keep the vehicle in the {lw} corner the whole time.",
    "Work the {lw} section only, and bring back the {c} object.",
    "Collect the {c} {n}, remaining within the {lw} area at all times.",
]
for i, template in enumerate(stay_quad_templates):
    for j, c in enumerate(COLORS):
        loc = LOCS[(i + j) % len(LOCS)]
        add(template.format(c=c, lw=LOC_WORDS[loc], n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", "unspecified", "collect_target", stay=loc)

# (c) stay_region = "pathway". The rules name a safe pathway but never define it
# geometrically, so it cannot be a compass value; the autonomy stack resolves the geometry.
pathway_sentences = [
    "Bring back the {c} {n} and do not leave the safe pathway.",
    "Keep to the marked corridor the whole way, the {c} target is the pickup.",
    "The {c} object comes home, and the vehicle stays on the designated path.",
    "Retrieve the {c} {n} without ever stepping off the laid-out route.",
    "Stay on the safe pathway. The {c} target is what we want.",
    "Fetch the {c} object -- the vehicle must not leave the marked lane.",
    # no target named, so the pathway constraint has to stand on its own wording
    "Do not leave the safe pathway at any point.",
    "The vehicle stays on the marked corridor this entire run.",
    "Keep to the designated path, no exceptions.",
    "Nothing off the laid-out route -- stay on it.",
    "You are restricted to the safe pathway.",
    "Hold the marked lane the whole way.",
]
for i, template in enumerate(pathway_sentences):
    c = COLORS[i % len(COLORS)]
    add(template.format(c=c, n=NOUNS[i % len(NOUNS)]),
        c if "{c}" in template else "unspecified",
        "none", "unspecified", "collect_target", stay="pathway")

# (d) containment WITH a release condition. until_region only ever appears here -- a keep-out
# region has no release condition in any rules example, so every avoid row keeps until=none.
# The pairs never share a compass component (S->E, not S->SE), so "which half is which" is
# never genuinely ambiguous, same discipline as PATH_WORD above.
UNTIL_PAIRS = [("S", "E"), ("N", "W"), ("E", "N"), ("W", "S")]
stay_until_templates = [
    "Stay to the far {sb} of the field until you reach the {uw} half, then collect the {c} {n}.",
    "Hold to the {sw} half until you make the {uw} side, after that you are free to move.",
    "Keep inside the {sw} half; once you reach the {uw} half that restriction lifts.",
    "Remain in the {sw} half of the course until you hit the {uw} half.",
    "Work the {sw} side only until you get to the {uw} half, then grab the {c} target.",
    "The {sw} half is your limit until you arrive at the {uw} half.",
    "Travel confined to the far {sb} until the {uw} half, then proceed normally.",
    "Do not leave the {sw} half before you reach the {uw} side of the field.",
]
for i, template in enumerate(stay_until_templates):
    for j, (sr, ur) in enumerate(UNTIL_PAIRS):
        c = COLORS[(i + j) % len(COLORS)]
        add(template.format(sw=HALF_WORDS[sr], sb=HALF_BARE[sr], uw=HALF_WORDS[ur],
                            c=c, n=NOUNS[(i + j) % len(NOUNS)]),
            c if "{c}" in template else "unspecified",
            "none", "unspecified", "collect_target", stay=sr, until=ur)

# (e) avoid AND stay in one command. This combination is the entire reason v3 split
# `constraints` into separate fields -- a single-valued enum could not hold both -- so if it
# has no training data the split bought nothing.
AVOID_STAY_PAIRS = [("SW", "N"), ("NE", "S"), ("NW", "E"), ("SE", "W")]
avoid_stay_templates = [
    "Stay in the {sw} half and keep clear of the {aw} corner.",
    "The {aw} quadrant is off limits; work the {sw} half only.",
    "Hold to the {sw} side of the field and treat the {aw} corner as a keep-out zone.",
    "Operate inside the {sw} half, and nothing enters the {aw} section.",
    "{AW} corner is closed. Remain in the {sw} half.",
    "Confine yourself to the {sw} half and route around the {aw} quadrant entirely.",
]
for i, template in enumerate(avoid_stay_templates):
    for j, (ar, sr) in enumerate(AVOID_STAY_PAIRS):
        add(template.format(sw=HALF_WORDS[sr], aw=LOC_WORDS[ar],
                            AW=LOC_WORDS[ar].capitalize()),
            "unspecified", "none", "unspecified", "collect_target", stay=sr, avoid=ar)

# (f) avoid_region WITH a named target. v11's 212 avoid rows ALL had target_color
# "unspecified", and on heldout5 the model missed every novel-wording avoid sentence that
# named a colour -- an unseen combination, not just unseen vocabulary. Also the first
# avoid_region rows stated in HALVES, which the schema has always allowed and no row used.
avoid_with_target = [
    ("Collect the {c} {n}, and keep out of the {aw} corner on the way.", "quad"),
    ("The {c} target is the pickup. The {aw} quadrant is restricted.", "quad"),
    ("Bring in the {c} object; nothing enters the {aw} section.", "quad"),
    ("Retrieve the {c} {n} and route around the {aw} corner, it is flagged.", "quad"),
    ("Get the {c} target. Stay out of the {hw} half of the field.", "half"),
    ("The {hw} half is off limits -- the {c} {n} still comes home.", "half"),
    ("Fetch the {c} object. No part of your route may cross into the {hw} side.", "half"),
    ("{HW} half is closed this round. The {c} target is your objective.", "half"),
]
for i, (template, kind) in enumerate(avoid_with_target):
    pool = LOCS if kind == "quad" else HALVES
    for j, c in enumerate(COLORS):
        r = pool[(i + j) % len(pool)]
        word = LOC_WORDS[r] if kind == "quad" else HALF_WORDS[r]
        add(template.format(c=c, n=NOUNS[(i + j) % len(NOUNS)],
                            aw=word, hw=word, HW=word.capitalize()),
            c, "avoid_regions", r, "collect_target")

# (g) MINIMAL PAIRS: route vs containment. Every sentence above that sets stay_region uses a
# containment verb (stay/remain/hold/confine/operate-in); the rows below use the same region
# word with a ROUTE verb (traverse/enter via/come across) and carry NO constraint. "Cut
# across the southern half" is a path, "stay to the far south" is a fence, and the only cue
# is the verb. distractor_route above already covers four of these; these add the half
# vocabulary that block never used.
#   NOTE on ratios: these land in collect_target/none (oversampled ~4x) while their partners
#   land in collect_target/stay (~2x), so balance() does NOT preserve the 1:1 authoring
#   ratio. Round 9 built machinery to pin pair ratios and it bought 2 examples out of 566, so
#   this is left alone deliberately -- but if stay_region starts firing on route sentences,
#   this skew is the first thing to check.
route_not_containment = [
    "Traverse the {hw} half on your way to the {c} {n} in the {lw} corner.",
    "Enter via the {hw} side, the {c} target is in the {lw} quadrant.",
    "Your approach runs through the {hw} half -- pick up the {c} object in the {lw} section.",
    "Come at it across the {hw} half; the {c} {n} sits in the {lw} corner.",
]
for i, template in enumerate(route_not_containment):
    for j, c in enumerate(COLORS):
        h = HALVES[(i + j) % len(HALVES)]
        loc = [l for l in LOCS if h not in l][(i + j) % 2]
        add(template.format(hw=HALF_WORDS[h], c=c, lw=LOC_WORDS[loc],
                            n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", loc, "collect_target")


# ---------- round 11: construction breadth for the fields v13 under-fired ----------
# v13 made 52 errors on 674 examples and they concentrate hard. Across ALL of them there were
# ZERO wrong-region errors -- the model never maps a region to the wrong compass value. Every
# region error was a trigger error: 19 stay_region missed, 4 spurious, 3 avoid_region missed,
# 5 spurious, 1 until_region missed. So the region vocabulary is solid and what is missing is
# recognising that a constraint was stated at all.
#
# This block is written from v13's heldout6 error dump, which SPENDS heldout6 as a clean
# instrument -- a later gain there partly measures paraphrase quality. That is the trade round
# 9 made by accident; heldout7 was written first, before any template below, to replace it.
#
# The failures were not missing verbs, they were missing CONSTRUCTIONS. Round 10 wrote eight
# containment templates and every one of them is an imperative with an explicit containment
# verb ("stay within", "do not leave"). v13 handled those and missed possessives ("your box
# for this attempt is the southern half"), exclusives ("east half only"), bare fragments
# ("South half. Nowhere else."), confirmations ("Is the north half my limit? Yes") and
# drift verbs ("don't wander out of"). Breadth below is by construction, not by synonym.

# (a) containment as POSSESSION / ASSIGNMENT -- the single biggest missed family
stay_possessive = [
    "The {hw} half is yours and nothing beyond it.",
    "{HW} half is your area of operations this round.",
    "You have the {hw} half, that is the lot.",
    "Your working envelope is the {hw} half.",
    "The {hw} half belongs to you; the rest does not.",
    "{HW} half is the whole of your ground today.",
]
# (b) containment as EXCLUSIVE / "only"
stay_exclusive = [
    "{HW} half only.",
    "The {hw} half, and only the {hw} half.",
    "Nowhere but the {hw} half.",
    "Strictly the {hw} half this attempt.",
    "{HW} half exclusively, no exceptions.",
]
# (c) containment as a BARE FRAGMENT with a negation
stay_fragment = [
    "{HW} half. Nothing past it.",
    "{HW} half -- hard edge.",
    "{HW} half, and not a wheel outside.",
    "{HW} half. That is the boundary.",
]
# (d) containment via DRIFT verbs (wander / stray / drift / creep)
stay_drift = [
    "Do not drift out of the {hw} half.",
    "No wandering past the {hw} half.",
    "Don't creep beyond the {hw} half.",
    "Nothing strays outside the {hw} half.",
]
# (e) containment via ENCLOSURE metaphors
stay_enclosure = [
    "You are walled into the {hw} half.",
    "Hemmed into the {hw} half for the duration.",
    "The {hw} half is sealed around you.",
    "Shut inside the {hw} half this run.",
]
# (f) containment as a CONFIRMATION or question-answer
stay_confirm = [
    "Can you leave the {hw} half? No.",
    "Am I clear that the {hw} half is the cap? You are.",
    "Is anything outside the {hw} half allowed? It is not.",
]
# (g) containment via LIMIT / CEILING nouns
stay_limit = [
    "The {hw} half is your limit.",
    "Your ceiling is the {hw} half.",
    "The {hw} half marks how far you go.",
    "Outer bound: the {hw} half.",
]
for group in (stay_possessive, stay_exclusive, stay_fragment, stay_drift,
              stay_enclosure, stay_confirm, stay_limit):
    for i, template in enumerate(group):
        for j, h in enumerate(HALVES):
            add(template.format(hw=HALF_WORDS[h], HW=HALF_WORDS[h].capitalize(),
                                hb=HALF_BARE[h]),
                "unspecified", "none", "unspecified", "collect_target", stay=h)

# (h) the same constructions on QUADRANTS and with a named target, because v13's worst
# heldout6 stratum was stay_quad at 33% -- when it missed the containment it spilled the
# quadrant into target_location, the round-3 "first direction wins" bias resurfacing.
stay_quad_round11 = [
    "The {lw} corner is yours and nothing beyond it -- {c} {n} is the pickup.",
    "{LW} section only. Bring in the {c} target.",
    "{LW} quadrant. Nothing past it. The {c} object comes home.",
    "Do not drift out of the {lw} corner; the {c} {n} is in there.",
    "You are walled into the {lw} section. Collect the {c} target.",
    "Your limit is the {lw} quadrant, and the {c} {n} is what we want.",
    "Can you leave the {lw} corner? No. {C} object, please.",
    "{LW} corner is your whole ground. The {c} {n} is the objective.",
]
for i, template in enumerate(stay_quad_round11):
    for j, c in enumerate(COLORS):
        loc = LOCS[(i + j) % len(LOCS)]
        add(template.format(c=c, C=c.capitalize(), lw=LOC_WORDS[loc],
                            LW=LOC_WORDS[loc].capitalize(), n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", "unspecified", "collect_target", stay=loc)

# (i) pathway in the same new constructions. v13 scored 50% on the pathway stratum with 12
# training rows, all of them explicit "do not leave"/"keep to" imperatives.
pathway_round11 = [
    "The marked lane is yours and nothing either side of it.",
    "Taped route only.",
    "The corridor. Nothing outside it.",
    "Do not drift off the safe pathway.",
    "You are walled into the marked corridor.",
    "Can you cut off the path? No.",
    "Your limit is the edge of the safe pathway.",
    "Designated route exclusively, no exceptions.",
    "On the lane, and not a wheel either side.",
    "The pathway is the boundary this run.",
]
for i, template in enumerate(pathway_round11):
    add(template, "unspecified", "none", "unspecified", "collect_target", stay="pathway")

# (j) avoid_region stated as a HALF. Round 10 gave this 16 rows and v13 missed "has been
# fenced off" and "is scratched" -- the same construction-breadth gap, on the other field.
# NOTE: heldout7's avoid_halves stratum was written first and uses "walled off", "write
# off", "keep every wheel out", "struck from the course", "barred", "forbidden" and
# "closed". A first draft of this list reused four of those and the overlap guard flagged
# them at 0.86-0.96 against heldout7 -- the contamination this round was supposed to avoid,
# caught by the check rather than by my reading. Vocabulary below is disjoint from it.
avoid_half_round11 = [
    "The {hw} half is a no-go for this run.",
    "{HW} half is out of play.",
    "Steer well away from the {hw} half.",
    "Nothing of yours enters the {hw} half.",
    "The {hw} half is suspended from the course.",
    "{HW} half: no entry.",
    "Give the {hw} half a wide berth.",
    "The {hw} half is fenced off for the round.",
    "Treat the {hw} half as a hazard zone.",
    "The {hw} half is unavailable to you.",
]
for i, template in enumerate(avoid_half_round11):
    for j, h in enumerate(HALVES):
        add(template.format(hw=HALF_WORDS[h], HW=HALF_WORDS[h].capitalize()),
            "unspecified", "avoid_regions", h, "collect_target")

# ---------- round 11: action boundaries ----------
# (k) pause vs abort MINIMAL PAIRS: identical opening, one modifier decides. 8 of v13's 23
# action errors were this pair, 4 each way, and the deciding word was always a modifier
# ("permanently", "temporarily", "briefly", "entirely").
#
# Writing them as pairs also fixes a ratio bug. pause had 35 distinct rows against abort's 62,
# so balance() oversampled pause 3.6x and abort 1.8x and a 1:1 authored pair came out 2:1 in
# the file. Round 9 built machinery to pin pair ratios and it bought 2 examples out of 566.
# Growing the pause pool is the cheaper lever than new machinery -- but see (o) below: the
# pairs alone did NOT equalise the factors, because they grow both pools at once.
PAUSE_ABORT_PAIRS = [
    ("Pens down.", "Pens down for good."),
    ("Take five.", "Take the rest of the day, we're out."),
    ("Idle it a moment.", "Kill it, we're through."),
    ("Park it briefly.", "Park it, run's over."),
    ("Breather -- back in a tick.", "Last call, we're finished."),
    ("Hold station briefly.", "Hold station, mission scrubbed."),
    ("Hang on, more to do.", "Hang on -- actually, bin the whole thing."),
    ("Wait one.", "Wait -- no, scrap it entirely."),
    ("Cool it for a minute.", "Cool it, the run is dead."),
    ("Sit idle, I'll wave you on.", "Sit idle, nothing more is coming."),
    ("Pause it there, not done.", "Pause it there -- permanently."),
    ("Rest a beat.", "Rest easy, that's the end of it."),
    ("Stop the clock, briefly.", "Stop the clock. We're done and staying dark."),
    ("Stand easy a second.", "Stand easy, the attempt is cancelled."),
]
for pause_text, abort_text in PAUSE_ABORT_PAIRS:
    add(pause_text, "unspecified", "none", "unspecified", "pause")
    add(abort_text, "unspecified", "none", "unspecified", "abort")

# (l) abort vs return_to_start MINIMAL PAIRS. v13 read three aborts as returns even though
# each said to STAY ("hold where you sit", "stay where you are"). The cue is movement: abort
# ends the run in place, return_to_start sends the vehicle home.
ABORT_RETURN_PAIRS = [
    ("Knock off and sit tight.", "Knock off and head for home."),
    ("Stop work, stay exactly there.", "Stop work and come back in."),
    ("Done -- plant it where it is.", "Done -- walk it back to the line."),
    ("Finish up and hold, stay hidden.", "Finish up and roll home empty."),
    ("Shut it down in place.", "Shut it down back at the start."),
    ("Cease and remain put.", "Cease and make your way back."),
    ("That's all -- no movement from you.", "That's all -- bring it in."),
    ("Stop there and stay dark.", "Stop there, then return to base."),
]
for abort_text, return_text in ABORT_RETURN_PAIRS:
    add(abort_text, "unspecified", "none", "unspecified", "abort")
    add(return_text, "unspecified", "none", "unspecified", "return_to_start")

# (m) resume vs retry MINIMAL PAIRS on the word "again". v13 read "Crack on." and "Spin it up
# again." as retry_send. "Again" alone resumes a held run; a retry has to name the operation
# being redone -- the scan, the tag, the code, the transmission.
RESUME_RETRY_TRIPLES = [
    ("Under way again.", "Scan it again.", "Send it again."),
    ("Green light, continue.", "Another pass on the tag.", "Push the code out once more."),
    ("Back in business.", "Re-read the chip.", "Re-transmit the numbers."),
    ("Resume from the hold.", "That read failed -- do it over.", "That send failed -- do it over."),
    ("Underway once more.", "Give the chip another try.", "Give the uplink another try."),
]
RESUME_RETRY_TRIPLES += [
    ("Pick it back up.", "The tag didn't register -- again.", "The code didn't land -- again."),
    ("You're live once more.", "Run the tag a second time.", "Run the uplink a second time."),
    ("Carry on from the hold.", "Have the reader try once more.", "Have the radio try once more."),
    ("Let's get moving.", "Redo the chip read.", "Redo the transmission."),
    ("Unpaused -- go.", "One more scan of the tag.", "One more push of the code."),
    ("Off the brakes, continue.", "Try the NFC read a second time.", "Try the send a second time."),
]
for resume_text, read_text, send_text in RESUME_RETRY_TRIPLES:
    add(resume_text, "unspecified", "none", "unspecified", "resume")
    add(read_text, "unspecified", "none", "unspecified", "retry_read")
    add(send_text, "unspecified", "none", "unspecified", "retry_send")

# (n) read_chip as a TERSE FRAGMENT. v13 read "Red cube. Chip only." and "Yellow cube in the
# northwest -- chip only." as collect_target: the fragment shape carries no verb, so the
# colour plus a noun looked like a retrieval. Those two sentences are held-out, so these use
# different frames.
chip_fragment = [
    "{C} {n} -- chip data only, leave the object.",
    "{C} target: tag read, no lift.",
    "{C} {n}. Numbers off it, nothing more.",
    "Data from the {c} object, that is all.",
    "{C} {n} -- read and walk away.",
    "Scan the {c} target. Do not pick it up.",
]
for i, template in enumerate(chip_fragment):
    for j, c in enumerate(COLORS):
        add(template.format(c=c, C=c.capitalize(), n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", "unspecified", "read_chip")


# (o) More wording for the three thinnest pools. At 1288 distinct rows, pause oversampled
# 3.6x, report_status 3.9x and avoid_stay 3.8x -- those cells are carried by a few dozen
# sentences copied four times, which is round 4's memorisation failure in miniature. "More
# wording, not more copies" is the fix that worked then.
#
# This is also where the pause/abort ratio note above gets corrected. I claimed the minimal
# pairs would bring pause's oversample factor down to abort's on their own. They did not:
# the pairs grew BOTH pools (pause 35->49, abort 62->84) so the ratio barely moved, 1:0.50
# to 1:0.52. The pause sentences below are deliberately UNPAIRED to close the gap, and even
# then it only reaches ~1:0.8, because the cell shares that set the ratio are a measured
# class-balance choice (round 4: held-out recall 0.762 at 12.5% share, 1.000 at 25%) and the
# pair ratio loses that argument. Documented rather than engineered around -- round 9 built
# machinery for exactly this and it bought 2 examples out of 566.
pause_extra = [
    "Hold up a sec.", "Give it a rest for now.", "Pencils down, briefly.",
    "On hold.", "Just wait there.",
    "Nothing for the moment.", "Hang fire.", "Take a knee.",
    "Keep still, I'll shout when.", "Pause -- stand by for more.", "Hold off, I'll be back to you.",
    "Halt for now, there's more coming.", "Wait where you are, briefly.",
    "Time out.", "Settle down a minute.", "Hold that thought.",
    "Don't move yet.", "Give me a second here.", "Stay there, I'm thinking.",
    "Not just now -- wait.", "Idle, pending my call.", "Steady -- hold.",
    "Freeze it, I'll release you.", "Hold the line a moment.",
    "Stop briefly, we continue after.", "Standby, resuming shortly.",
    "Suspend briefly, more instructions coming.", "Wait up, not finished here.",
    "Hold position, temporary only.", "Brief hold, then we go on.",
]
for t in pause_extra:
    add(t, "unspecified", "none", "unspecified", "pause")

abort_extra = [
    "Scrub the attempt, stay where you sit.", "Call it off and keep low.",
    "That's the run gone -- hold and stay quiet.", "Terminate. No movement.",
    "We're pulling the plug, stop in place.", "Run's void, remain concealed.",
    "End it here and don't be seen.", "Null the attempt, freeze.",
    "Binning this one -- stay down.", "The attempt is dead, hold still.",
    "Stand down for good, stay hidden.", "Shut it down, nothing further.",
]
for t in abort_extra:
    add(t, "unspecified", "none", "unspecified", "abort")

report_extra = [
    "Position and task, please.", "Sitrep, when you get a chance.", "Which part of the job are you on?",
    "Talk to me -- what are you on?", "Tell me what you're doing right now.",
    "State your current task.", "How's it going out there?",
    "I need an update on your progress.",
]
for t in report_extra:
    add(t, "unspecified", "none", "unspecified", "report_status")

# (p) avoid + stay together, the combination that justified splitting the field. 24 distinct
# rows oversampled 3.8x, and heldout6's avoid_stay stratum scored 62.5%. Same constructions
# as the round-11 containment families above, so the two constraints are stated in one breath.
avoid_stay_round11 = [
    "The {sw} half is yours and the {aw} corner is a no-go.",
    "{SW} half only, and nothing enters the {aw} quadrant.",
    "{SW} half. Nothing past it. The {aw} corner is out of play.",
    "Do not drift out of the {sw} half, and give the {aw} section a wide berth.",
    "You are walled into the {sw} half; the {aw} corner is fenced off.",
    "Your limit is the {sw} half and the {aw} quadrant is a hazard zone.",
    "Can you leave the {sw} half? No. And the {aw} corner is shut.",
    "{SW} half is your ground. Steer well away from the {aw} section.",
    "Stay to the marked corridor and keep out of the {aw} quadrant.",
    "The pathway is your bound, and the {aw} corner is suspended.",
]
for i, template in enumerate(avoid_stay_round11):
    for j, (ar, sr) in enumerate(AVOID_STAY_PAIRS):
        stay = "pathway" if "corridor" in template or "pathway" in template else sr
        word = HALF_WORDS[sr]
        add(template.format(sw=word, SW=word.capitalize(), aw=LOC_WORDS[ar],
                            AW=LOC_WORDS[ar].capitalize()),
            "unspecified", "none", "unspecified", "collect_target", stay=stay, avoid=ar)


# ---------- round 12: polarity. Which way does the fence face? ----------
# v14's failure mode is not v13's. v13 UNDER-FIRED, saying "none" when a constraint was
# stated. v14 finds the constraint and reads the region off the sentence correctly, then puts
# it in the WRONG FIELD, in both directions:
#   "Anything outside the south half is out of bounds."  stay=S  -> avoid=S
#   "The west half has been struck from the course."     avoid=W -> stay=W
# Across all 740 examples v14 made exactly 1 wrong-region error and 31 trigger/slot errors.
# So region extraction is solved and POLARITY is not.
#
# This is self-inflicted, and visible by auditing the training set alone -- no test data
# needed. Rounds 10 and 11 handed both fields enclosure metaphors drawn from one vocabulary
# and never contrasted them: "seal" appears on 8 avoid rows and 4 stay rows, "fenc" 8 and 4,
# "limit" 8 and 21, "close" 8 and 4. Near-synonyms split across opposite fields with nothing
# teaching the model which way each points.
#
# The fix is minimal pairs on polarity alone: same frame, same region, one word flipped. If
# the model keys on vocabulary it must get one side of every pair wrong, so the pairs cannot
# be satisfied by memorising words.
#
# Note on instruments: heldout8 was written BEFORE this block and tests complement
# constructions ("everywhere except X", "all but X", "leaving vs entering"). Deliberately
# none of those realisations appear here -- this block uses in/out, leave/enter, inside/
# outside and "the only place / the one place". The 0.85 guard enforces it.

# (a) the canonical pair: one preposition decides the field
inout_pairs = [
    ("Stay in the {hw} half.",                  "Stay out of the {hw} half."),
    ("Keep inside the {hw} half.",              "Keep outside the {hw} half."),
    ("Work inside the {hw} half.",              "Work clear of the {hw} half."),
    ("You belong in the {hw} half.",            "You do not belong in the {hw} half."),
    ("The {hw} half is where you may go.",      "The {hw} half is where you may not go."),
    ("The only place you may be is the {hw} half.",
     "The one place you may not be is the {hw} half."),
]
for stay_t, avoid_t in inout_pairs:
    for h in HALVES:
        w = HALF_WORDS[h]
        add(stay_t.format(hw=w), "unspecified", "none", "unspecified",
            "collect_target", stay=h)
        add(avoid_t.format(hw=w), "unspecified", "avoid_regions", h, "collect_target")

# (b) leave vs enter. Round 10 taught "Do not leave the {hw} half" as containment and never
# taught its mirror, so "enter" had to be inferred from unrelated avoid wording.
leave_enter_pairs = [
    ("Do not exit the {hw} half.",              "Do not enter the {hw} half."),
    ("Leaving the {hw} half is not allowed.",   "Entering the {hw} half is not allowed."),
    ("No part of your route leaves the {hw} half.",
     "No part of your route enters the {hw} half."),
    ("Crossing out of the {hw} half is a fault.",
     "Crossing into the {hw} half is a fault."),
]
for stay_t, avoid_t in leave_enter_pairs:
    for h in HALVES:
        w = HALF_WORDS[h]
        add(stay_t.format(hw=w), "unspecified", "none", "unspecified",
            "collect_target", stay=h)
        add(avoid_t.format(hw=w), "unspecified", "avoid_regions", h, "collect_target")

# (c) the same polarity contrast on QUADRANTS and with a target named, because v14's
# target_location spill happened on exactly these rows: when it misread the polarity it
# sometimes dropped the quadrant into target_location instead of either constraint field.
quad_polarity_pairs = [
    ("Stay in the {lw} corner. The {c} {n} is the pickup.",
     "Stay out of the {lw} corner. The {c} {n} is the pickup."),
    ("Keep inside the {lw} section, and bring in the {c} target.",
     "Keep clear of the {lw} section, and bring in the {c} target."),
    ("{C} object, and the {lw} quadrant is where you may work.",
     "{C} object, and the {lw} quadrant is where you may not work."),
    ("Do not exit the {lw} corner; the {c} {n} comes home.",
     "Do not enter the {lw} corner; the {c} {n} comes home."),
]
for i, (stay_t, avoid_t) in enumerate(quad_polarity_pairs):
    for j, c in enumerate(COLORS):
        loc = LOCS[(i + j) % len(LOCS)]
        kw = dict(c=c, C=c.capitalize(), lw=LOC_WORDS[loc], n=NOUNS[(i + j) % len(NOUNS)])
        add(stay_t.format(**kw), c, "none", "unspecified", "collect_target", stay=loc)
        add(avoid_t.format(**kw), c, "avoid_regions", loc, "collect_target")

# (d) pathway polarity. "On the path" is containment; "off the path" is not a keep-out
# region, it is the same containment stated negatively -- both are stay=pathway. This pair
# exists to stop the model reading "off"/"not" as an avoid cue by itself.
pathway_polarity = [
    "Stay on the marked path.",
    "Do not stray off the marked path.",
    "On the corridor at all times.",
    "Never off the corridor.",
    "Inside the taped lane, always.",
    "Not once outside the taped lane.",
]
for t in pathway_polarity:
    add(t, "unspecified", "none", "unspecified", "collect_target", stay="pathway")

# ---------- round 12: the stop-verb cluster ----------
# 15 of v14's 19 action errors sit here: abort->pause 4, abort->return_to_start 3, and six
# ways of mishandling resume. Every one turns on a cue that is NOT the verb -- the verb is
# shared. So the verb is held constant and only the cue varies, three ways:
#   pause            expects to resume        -> a time-limited modifier
#   abort            ends the run in place    -> finality AND/OR concealment, no movement
#   return_to_start  ends the run by coming home -> an explicit movement-to-base cue
# Round 11 did pause/abort and abort/return as separate two-way pairs. Making it one
# three-way frame per verb is the change: the model sees all three readings of the same
# opening words, which is the only way the cue can be the thing it learns.
STOP_TRIPLES = [
    "Break off",
    "Pack up",
    "Wind it down",
    "Call time",
    "Ease up",
    "Pull up",
    "Shelve it",
    "Put it down",
]
for verb in STOP_TRIPLES:
    add(f"{verb} for a moment, I'll wave you on.",
        "unspecified", "none", "unspecified", "pause")
    add(f"{verb} where you stand and stay out of sight -- that's the attempt.",
        "unspecified", "none", "unspecified", "abort")
    add(f"{verb} and drive yourself back to the start.",
        "unspecified", "none", "unspecified", "return_to_start")

# (f) resume, with no operation named. v14 read resume as pause twice and as retry_send
# twice. A retry names the thing being redone; a bare "go again" does not.
resume_extra = [
    "Back to work.", "Onwards.", "You may proceed.", "Resume the task.",
    "Clear to continue.", "Carry on where you were.", "Moving again, please.",
    "Hold's over -- go.", "Release -- continue the job.", "Pick up the task again.",
]
for t in resume_extra:
    add(t, "unspecified", "none", "unspecified", "resume")

# retry_read is the one cell still hitting the 4.0x oversample cap -- 29 distinct sentences
# copied four times. More wording, not more copies.
retry_read_extra = [
    "The chip read came back empty, go again.",
    "Scan failed. Repeat it.",
    "Didn't catch the tag -- once more.",
    "Bad read. Do that scan over.",
    "The NFC didn't take, try it again.",
    "Reread that chip for me.",
    "That tag scan was no good, repeat.",
    "Take another run at the chip read.",
]
for t in retry_read_extra:
    add(t, "unspecified", "none", "unspecified", "retry_read")

# (g) read_chip in a terse frame, still 2 errors in v14 and 3 in v13. The fragment shape
# carries no verb, so a colour plus a noun reads as a retrieval.
chip_extra = [
    "{C} {n}, chip only.",
    "{C} target -- the code, not the object.",
    "Only the chip off the {c} {n}.",
    "{C} object: read it where it sits.",
]
for i, template in enumerate(chip_extra):
    for j, c in enumerate(COLORS):
        add(template.format(c=c, C=c.capitalize(), n=NOUNS[(i + j) % len(NOUNS)]),
            c, "none", "unspecified", "read_chip")


# v3 note: the cell key used to be (action, constraints), and `constraints` is gone. The
# replacement collapses the three region fields to WHICH KIND of constraint is present,
# which is what the balancing was ever about -- the specific region is already balanced by
# the templates looping over all of LOCS/HALVES.
def cell_kind(e):
    avoid, stay = e["avoid_region"] != "none", e["stay_region"] != "none"
    return {(False, False): "none", (True, False): "avoid",
            (False, True): "stay", (True, True): "avoid_stay"}[(avoid, stay)]


# Round 4-8 shares, which converged at 97.70% on five sets (v11). Reproduced verbatim so
# the v3 cells can be added WITHOUT re-guessing ten numbers that are known to work.
V11_CELLS = {
    ("collect_target", "none"): 0.26,
    ("collect_target", "avoid"): 0.16,
    ("read_chip", "none"): 0.14,
    ("abort", "none"): 0.08,
    ("return_to_start", "none"): 0.07,
    ("report_status", "none"): 0.06,
    ("pause", "none"): 0.09,
    ("resume", "none"): 0.06,
    # retry_* sit below the 10% line on purpose: their wording ("again", "re-") is the most
    # distinctive in the schema, so they should not need the share. Watch them on heldout3.
    ("retry_read", "none"): 0.05,
    ("retry_send", "none"): 0.05,
}

# The two new v3 cells. 0.15 total is a deliberate guess, reasoned from avoid_region: it
# recalled 0.762 at 12.5% share and 1.000 at 25%, so ~11% is the lowest share that has ever
# worked for a constraint class, and avoid_stay gets less because co-occurrence is the rarer
# phrasing in the rules. until_region is NOT its own cell -- it only ever appears inside a
# stay row, and splitting it would halve the pool each sub-cell draws from.
STAY_CELLS = {
    ("collect_target", "stay"): 0.11,
    ("collect_target", "avoid_stay"): 0.04,
}

# Everything from v11 keeps its RELATIVE proportion and is scaled down to make room, so the
# only intentional change between v11 and v13 is the new classes, not a reshuffle of the old
# ones. ponytail: computed, not a hand-retyped table -- a table drifts from its own comment.
# (V11_CELLS sums to 1.02, not 1.0 -- a pre-existing wart. balance() anchors on whichever
# cell is closest to target and normalises, so only the RATIOS matter; the scale below keeps
# the grand total put so a share here still reads as roughly its percentage of the set.)
_v11_total = sum(V11_CELLS.values())
_scale = (_v11_total - sum(STAY_CELLS.values())) / _v11_total
TARGET_CELLS = {cell: share * _scale for cell, share in V11_CELLS.items()}
TARGET_CELLS.update(STAY_CELLS)
assert abs(sum(TARGET_CELLS.values()) - _v11_total) < 1e-9

# A cell oversampled past this is memorising a handful of sentences rather than learning the
# class; the builder warns and stops at the cap instead of silently producing 8x duplicates.
MAX_OVERSAMPLE = 4.0


def balance(rows):
    """Oversample toward TARGET_CELLS on the joint (action, constraint-kind) cell.

    Deterministic round-robin over each cell pool, so a rebuild reproduces the file.
    Duplicates are the point: they reweight the loss without inventing sentences.
    ponytail: oversample rather than discard -- dropping real sentences to hit a ratio
    would throw away coverage already paid for.
    """
    import collections
    pools = collections.defaultdict(list)
    for r in rows:
        e = r["expected"]
        pools[(e["action"], cell_kind(e))].append(r)

    unexpected = set(pools) - set(TARGET_CELLS)
    if unexpected:
        raise SystemExit(f"cell with no target, refusing to guess: {sorted(unexpected)}")

    # TRIED AND REJECTED (v16): splitting the avoid+stay budget in proportion to pool size.
    # Round 12's polarity pairs reach the model skewed -- avoid oversampled 1.63x against
    # stay's 1.00x, so an authored 1:1 pair arrives 1:0.61 -- and the swaps were directional
    # the same way, so the skew looked causal. Equalising it is easy: give the two cells their
    # COMBINED share split by pool size and their want/have ratio is equal by construction.
    # That was built, verified (ratio exactly 1:1.00, both cells 1.00x, 240 avoid / 316 stay
    # distinct rows) and trained as v16. It did not work:
    #
    #   polarity swaps          12 -> 13     the hypothesis, refuted
    #   stay MISSED / SPURIOUS  16/12 -> 13/15
    #   avoid MISSED / SPURIOUS 10/8  -> 6/10
    #   pooled, 9 sets          92.80% -> 93.06%   (+0.25, all of it heldout6)
    #   pooled, minus heldout6  94.29% -> 93.61%   (-0.68)
    #   original coverage sets  97.17% -> 96.29%   (-0.88, 9 newly broken vs 4 fixed)
    #
    # What it actually did was move the firing threshold, not polarity discrimination: it
    # traded MISSED for SPURIOUS on both fields and left the swap count alone. Raising stay's
    # share 11% -> 14% made it over-fire on sentences that merely mention a direction ("Run
    # the western edge on the way out" -> stay=W), and it cost core competence on easy rows
    # ("Fetch the yellow block." -> read_chip). The only gain was +12.5% on heldout6, which
    # round 11 was written from and which is therefore partly a paraphrase-memory score.
    #
    # So the hand-set shares stay. They were tuned on measured held-out recall (round 4:
    # 0.762 at 12.5% share, 1.000 at 25%) and that evidence outranks a pair-ratio argument.
    # Polarity is orthogonal to class balance and needs wording this author did not generate.
    # anchor on whichever cell is already closest to its target, so nothing shrinks
    implied_total = max(len(pools[c]) / TARGET_CELLS[c] for c in pools)

    out, capped = list(rows), []
    for cell, share in TARGET_CELLS.items():
        pool = pools.get(cell)
        if not pool:
            raise SystemExit(f"target cell {cell} has no examples")
        want, have = round(implied_total * share), len(pool)
        if want / have > MAX_OVERSAMPLE:
            capped.append((cell, have, want, round(have * MAX_OVERSAMPLE)))
            want = round(have * MAX_OVERSAMPLE)
        out.extend(pool[i % have] for i in range(max(0, want - have)))
    for cell, have, want, got in capped:
        print(f"NOTE: {cell} capped at {MAX_OVERSAMPLE}x -- {have} distinct -> {got}, "
              f"target wanted {want}. Write more sentences for this cell.")
    return out


def main():
    # the held-out set is checked too: it is the only eval set not yet used for model
    # selection, and training on its wording would destroy the one clean instrument left
    # ALL held-out sets, not just eval+heldout. The two-set version could not see
    # heldout2/3/4/5, and templates written against an observed heldout3 or heldout4 failure
    # are exactly the case that needs checking -- a draft of round 9 put 8 verbatim test
    # sentences into training and a hand-rolled check still missed one plus a 0.918
    # near-duplicate. Cost is a slower difflib pass over 6 sets; worth it.
    sources = {name: f"data/{name}_set.jsonl" for name in
               ("eval", "heldout", "heldout2", "heldout3", "heldout4", "heldout5",
                "heldout6", "heldout7", "heldout8")}
    existing = {}
    for name, path in sources.items():
        try:
            with open(path, encoding="utf-8") as f:
                existing[name] = {json.loads(line)["text"] for line in f}
        except FileNotFoundError:
            print(f"note: {path} not found, skipping its overlap check")

    for name, texts in existing.items():
        overlap = sorted({r["text"] for r in rows} & texts)
        if overlap:
            raise SystemExit(f"train/{name} overlap detected, aborting write: {overlap}")

    # near-duplicates matter as much as exact hits once we are writing templates against
    # observed failures -- a paraphrase of a held-out sentence measures recall, not skill
    import difflib
    for name, texts in existing.items():
        texts = list(texts)
        worst = []
        for r in {r["text"] for r in rows}:
            m = max(texts, key=lambda t: difflib.SequenceMatcher(None, r, t).ratio())
            ratio = difflib.SequenceMatcher(None, r, m).ratio()
            if ratio >= 0.85:
                worst.append((ratio, r, m))
        if worst:
            print(f"WARNING: {len(worst)} train sentence(s) >=0.85 similar to {name}:")
            for ratio, r, m in sorted(worst, reverse=True)[:10]:
                print(f"  {ratio:.3f}")
                print(f"    TRAIN  : {r}")
                print(f"    {name.upper():7s}: {m}")

    # "distinct" below counts AUTHORED rows, which is not the same as distinct sentences: a
    # sentence written twice in two different template lists is two rows and gets double
    # weight. v15 trained with 3 such pairs ("Back to work.", "Resume the task.", "How's it
    # going out there?") -- all three agreed on their label, so this is weighting, not a
    # labelling conflict, and 3 of 1554 is why it is reported rather than silently deduped:
    # deduping here would change the output file and break reproducibility of a measured run.
    # Conflicting labels WOULD be a real bug, so that case is fatal.
    import collections as _c
    text_counts = _c.Counter(r["text"] for r in rows)
    dups = {t: n for t, n in text_counts.items() if n > 1}
    if dups:
        conflicting = {}
        for t in dups:
            labels = [r["expected"] for r in rows if r["text"] == t]
            if any(x != labels[0] for x in labels):
                conflicting[t] = labels
        if conflicting:
            raise SystemExit(f"same sentence, different labels: {conflicting}")
        print(f"NOTE: {len(rows)} authored rows but {len(text_counts)} distinct sentences; "
              f"{len(dups)} written twice (same label, so double weight only): "
              + ", ".join(repr(t) for t in dups))

    balanced = balance(rows)

    import collections
    with open("data/train_set.jsonl", "w", encoding="utf-8") as f:
        for r in balanced:
            f.write(json.dumps(r) + "\n")
    cmix = collections.Counter(cell_kind(r["expected"]) for r in balanced)
    amix = collections.Counter(r["expected"]["action"] for r in balanced)
    n = len(balanced)
    print(f"wrote {n} examples to data/train_set.jsonl "
          f"({len(rows)} distinct + {n - len(rows)} oversampled), 0 overlap with eval/heldout")
    print("  constraint kind: " + "  ".join(f"{k}={v} ({v/n:.1%})" for k, v in cmix.most_common()))
    print("  action:      " + "  ".join(f"{k}={v} ({v/n:.1%})" for k, v in amix.most_common()))


if __name__ == "__main__":
    main()
