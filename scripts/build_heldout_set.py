"""Builder for data/heldout_set.jsonl -- a fresh held-out set written to settle
v3 vs v5, which data/eval_set.jsonl can no longer do: that set has been used for
model selection three times (round 2 -> v3, then v4 and v5), so it is an
optimistic estimate for v5 in particular.

Every sentence here is new wording, verified against BOTH data/train_set.jsonl and
data/eval_set.jsonl (exact + near-duplicate) before writing.

Sentences are UNIQUE, never slot-rotations of one frame: four colour-swaps of the
same sentence are highly correlated, so they would inflate n without adding
independent evidence and make McNemar report false confidence.

STRATA, scored separately -- an average would hide a model-specific bias:
  familiar  - families BOTH v3 and v5 trained on. Should be a tie; v5 losing here
              means the round-3 templates eroded something.
  contested - the families v5's round-3 additions targeted (permission-framed
              avoid_objects, directional distractors, proximity-to-objects,
              generic-obstacle negatives), in wording neither model has seen.
              This is the real test of whether v5's gain generalises or memorised.
  novel     - syntactic shapes neither model trained on (questions, passive voice,
              terse fragments, subordinate clauses, indirect requests).
"""
import collections
import difflib
import json

from schema import expand_v2

# label tuple = (target_color, constraints, target_location, action)

FAMILIAR = [
    # --- avoid_objects, canonical prohibition wording ---
    ("Bring the blue die in and don't make contact with anything else out there.", ("blue", "avoid_objects", "unspecified", "collect_target")),
    ("The red cube is the pickup -- everything else stays untouched.", ("red", "avoid_objects", "unspecified", "collect_target")),
    ("Haul the yellow block back without brushing the other pieces.", ("yellow", "avoid_objects", "unspecified", "collect_target")),
    ("Black target only. Do not touch the rest.", ("black", "avoid_objects", "unspecified", "collect_target")),
    ("Collect the blue object and keep the other dice exactly as they are.", ("blue", "avoid_objects", "unspecified", "collect_target")),
    ("Take the red die home, nothing else on that field gets moved.", ("red", "avoid_objects", "unspecified", "collect_target")),
    ("Yellow is the pickup; the remaining objects must not be touched.", ("yellow", "avoid_objects", "unspecified", "collect_target")),
    ("Get the black cube without laying a finger on the others.", ("black", "avoid_objects", "unspecified", "collect_target")),
    ("Retrieve the blue block, and no contact with the rest of the field.", ("blue", "avoid_objects", "unspecified", "collect_target")),
    ("Red object comes in clean -- do not disturb the others.", ("red", "avoid_objects", "unspecified", "collect_target")),
    ("Pull the yellow die out without shifting any of the other objects.", ("yellow", "avoid_objects", "unspecified", "collect_target")),
    ("Black cube is the objective, everything else is hands off.", ("black", "avoid_objects", "unspecified", "collect_target")),
    ("Grab the blue target from the northwest and leave the others alone.", ("blue", "avoid_objects", "NW", "collect_target")),
    ("Red die in the southeast -- take it without touching the rest.", ("red", "avoid_objects", "SE", "collect_target")),
    ("Collect the yellow object in the northeast, no contact with the other pieces.", ("yellow", "avoid_objects", "NE", "collect_target")),
    ("Black block sits in the southwest; bring it in without disturbing anything.", ("black", "avoid_objects", "SW", "collect_target")),

    # --- plain retrieve, no location slot ---
    ("Go get the blue die and bring it home.", ("blue", "none", "unspecified", "collect_target")),
    ("The red object is the target -- collect it.", ("red", "none", "unspecified", "collect_target")),
    ("Bring the yellow cube in.", ("yellow", "none", "unspecified", "collect_target")),
    ("Black block is what we want. Go fetch it.", ("black", "none", "unspecified", "collect_target")),
    ("Your pickup is the blue target.", ("blue", "none", "unspecified", "collect_target")),
    ("Head out and collect the red die.", ("red", "none", "unspecified", "collect_target")),
    ("We want the yellow object retrieved.", ("yellow", "none", "unspecified", "collect_target")),
    ("Go after the black cube and return with it.", ("black", "none", "unspecified", "collect_target")),
    ("Blue is the objective this run.", ("blue", "none", "unspecified", "collect_target")),
    ("Collect the red block and come home.", ("red", "none", "unspecified", "collect_target")),
    ("The yellow die is yours to bring back.", ("yellow", "none", "unspecified", "collect_target")),
    ("Go recover the black object.", ("black", "none", "unspecified", "collect_target")),
    ("Pick up the red cube and head for the start.", ("red", "none", "unspecified", "collect_target")),
    ("The yellow target needs collecting.", ("yellow", "none", "unspecified", "collect_target")),
    ("Blue object -- retrieve and return.", ("blue", "none", "unspecified", "collect_target")),
    ("Bring the black target back to the starting zone.", ("black", "none", "unspecified", "collect_target")),

    # --- retrieve with a location slot ---
    ("The blue die is in the northwest. Go get it.", ("blue", "none", "NW", "collect_target")),
    ("Red cube, southeast corner -- collect it.", ("red", "none", "SE", "collect_target")),
    ("You'll find the yellow object in the northeast.", ("yellow", "none", "NE", "collect_target")),
    ("Black block is sitting in the southwest quadrant.", ("black", "none", "SW", "collect_target")),
    ("Our blue piece is down in the southeast somewhere.", ("blue", "none", "SE", "collect_target")),
    ("The red die is over in the northwest area.", ("red", "none", "NW", "collect_target")),
    ("Pick up the yellow cube in the southwest.", ("yellow", "none", "SW", "collect_target")),
    ("Black object, northeast quarter. Bring it in.", ("black", "none", "NE", "collect_target")),
    ("Head northwest and collect the blue block.", ("blue", "none", "NW", "collect_target")),
    ("The red target waits in the southeast.", ("red", "none", "SE", "collect_target")),
    ("Yellow die is in the northeast corner -- retrieve it.", ("yellow", "none", "NE", "collect_target")),
    ("Go to the southwest and pick up the black cube.", ("black", "none", "SW", "collect_target")),
    ("Blue object is positioned in the northeast.", ("blue", "none", "NE", "collect_target")),
    ("Retrieve the red block from the southwest corner.", ("red", "none", "SW", "collect_target")),
    ("The yellow target is in the northwest quadrant.", ("yellow", "none", "NW", "collect_target")),
    ("Black die, southeast side. Go collect it.", ("black", "none", "SE", "collect_target")),

    # --- object relocated ---
    ("Be advised: the blue die shifted to the southwest.", ("blue", "none", "SW", "update_only")),
    ("Red target has been repositioned to the northeast.", ("red", "none", "NE", "update_only")),
    ("The yellow cube is no longer where it was; check the northwest.", ("yellow", "none", "NW", "update_only")),
    ("Black object moved -- southeast quadrant now.", ("black", "none", "SE", "update_only")),
    ("New location on the blue target: northeast corner.", ("blue", "none", "NE", "update_only")),
    ("The red die got bumped over to the southwest.", ("red", "none", "SW", "update_only")),
    ("Yellow object is now in the southeast section.", ("yellow", "none", "SE", "update_only")),
    ("Black cube has shifted to the northwest quarter.", ("black", "none", "NW", "update_only")),
    ("Updated position, blue block: southwest side.", ("blue", "none", "SW", "update_only")),
    ("The red target now sits in the northwest.", ("red", "none", "NW", "update_only")),
    ("Yellow die relocated to the northeast area.", ("yellow", "none", "NE", "update_only")),
    ("Black object's new spot is the southeast corner.", ("black", "none", "SE", "update_only")),

    # --- abort ---
    ("Shut it all down and don't move a wheel.", ("unspecified", "none", "unspecified", "abort")),
    ("We're aborting. Stay silent and stay still.", ("unspecified", "none", "unspecified", "abort")),
    ("End the attempt right now, stay concealed.", ("unspecified", "none", "unspecified", "abort")),
    ("Stop. No movement, no transmission.", ("unspecified", "none", "unspecified", "abort")),
    ("Scrap this run and hold where you sit.", ("unspecified", "none", "unspecified", "abort")),
    ("Everything halts here -- remain hidden.", ("unspecified", "none", "unspecified", "abort")),
    ("The attempt is cancelled. Freeze.", ("unspecified", "none", "unspecified", "abort")),
    ("Power it down and stay out of view.", ("unspecified", "none", "unspecified", "abort")),
    ("We've lost the window. Abort and hold.", ("unspecified", "none", "unspecified", "abort")),
    ("No more advancing, the run is dead.", ("unspecified", "none", "unspecified", "abort")),
    ("Cease movement immediately and lie low.", ("unspecified", "none", "unspecified", "abort")),
    ("Kill it here. Nothing moves.", ("unspecified", "none", "unspecified", "abort")),
    ("Pack it in -- stop and stay dark.", ("unspecified", "none", "unspecified", "abort")),
    ("That's a scratch. Hold position, stay quiet.", ("unspecified", "none", "unspecified", "abort")),
    ("Stop the run and remain undetected.", ("unspecified", "none", "unspecified", "abort")),
    ("Abort the attempt, keep the system concealed.", ("unspecified", "none", "unspecified", "abort")),

    # --- return_to_start (no target object named) ---
    ("Just come home, nothing to carry.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Bring the vehicle in, we're not collecting.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Drive back to the start and stop there.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Return to the launch box, empty.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Head for the starting zone, leave the field alone.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("We're resetting -- get the system back to start.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Come back in. No pickup.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Bring it home without anything.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Return the vehicle to the start, that's all.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Roll back to the starting square.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Make your way home, the run's over.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Get the rover back to base, nothing to bring.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Come back to the launch point now.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("We're pulling it in. Back to start.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Return to the start area and wait.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Bring the system in, we're finished searching.", ("unspecified", "none", "unspecified", "return_to_start")),

    # --- read_chip, no location slot ---
    ("Read the tag on the blue die and call it in.", ("blue", "none", "unspecified", "read_chip")),
    ("We need the chip number off the red cube.", ("red", "none", "unspecified", "read_chip")),
    ("Scan the yellow object's tag, don't lift it.", ("yellow", "none", "unspecified", "read_chip")),
    ("Take a chip reading from the black block.", ("black", "none", "unspecified", "read_chip")),
    ("Tag data on the blue target, that's all.", ("blue", "none", "unspecified", "read_chip")),
    ("Get the code off the red die and stay there.", ("red", "none", "unspecified", "read_chip")),
    ("Scan the yellow cube and report.", ("yellow", "none", "unspecified", "read_chip")),
    ("Chip read on the black object, leave it in place.", ("black", "none", "unspecified", "read_chip")),
    ("We want the tag ID from the blue block.", ("blue", "none", "unspecified", "read_chip")),
    ("Read the red target's chip and hold.", ("red", "none", "unspecified", "read_chip")),
    ("Pull the code from the yellow die.", ("yellow", "none", "unspecified", "read_chip")),
    ("Scan the black cube's tag and send it through.", ("black", "none", "unspecified", "read_chip")),

    # --- read_chip with a location slot ---
    ("Read the chip on the blue die in the northwest.", ("blue", "none", "NW", "read_chip")),
    ("Tag scan on the red cube over in the southeast.", ("red", "none", "SE", "read_chip")),
    ("Get the chip code from the yellow object in the northeast.", ("yellow", "none", "NE", "read_chip")),
    ("Scan the black block sitting in the southwest.", ("black", "none", "SW", "read_chip")),
    ("Chip data from the blue target in the southeast corner.", ("blue", "none", "SE", "read_chip")),
    ("Read the tag on the red die in the northwest quadrant.", ("red", "none", "NW", "read_chip")),
    ("Tag reading on the yellow cube in the southwest.", ("yellow", "none", "SW", "read_chip")),
    ("Scan the black object's chip in the northeast section.", ("black", "none", "NE", "read_chip")),

    # --- avoid_regions ---
    ("The northwest is closed for this attempt.", ("unspecified", "avoid_regions", "NW", "collect_target")),
    ("Don't route through the southeast quarter.", ("unspecified", "avoid_regions", "SE", "collect_target")),
    ("Northeast corner is off the table now.", ("unspecified", "avoid_regions", "NE", "collect_target")),
    ("Keep the vehicle out of the southwest.", ("unspecified", "avoid_regions", "SW", "collect_target")),
    ("That northwest section is barred.", ("unspecified", "avoid_regions", "NW", "collect_target")),
    ("Nothing goes through the southeast region.", ("unspecified", "avoid_regions", "SE", "collect_target")),
    ("Stay outside the northeast quadrant.", ("unspecified", "avoid_regions", "NE", "collect_target")),
    ("The southwest area is restricted from here.", ("unspecified", "avoid_regions", "SW", "collect_target")),
    ("Hold off the northwest quarter entirely.", ("unspecified", "avoid_regions", "NW", "collect_target")),
    ("Southeast is a no-entry zone this run.", ("unspecified", "avoid_regions", "SE", "collect_target")),
    ("Skip the northeast section on your route.", ("unspecified", "avoid_regions", "NE", "collect_target")),
    ("The southwest corner is blocked off.", ("unspecified", "avoid_regions", "SW", "collect_target")),
    ("Judges closed the northwest region.", ("unspecified", "avoid_regions", "NW", "collect_target")),
    ("Do not pass through the southeast area.", ("unspecified", "avoid_regions", "SE", "collect_target")),
    ("Northeast quarter is out of play.", ("unspecified", "avoid_regions", "NE", "collect_target")),
    ("Keep well outside the southwest section.", ("unspecified", "avoid_regions", "SW", "collect_target")),
]

# The families v5's round-3 templates targeted, in wording neither model has seen.
# If v5's eval gain was memorisation, it shows up here as no advantage.
CONTESTED = [
    # --- avoid_objects framed as PERMISSION granted to the target, not prohibition ---
    ("You may handle the blue die and nothing besides it.", ("blue", "avoid_objects", "unspecified", "collect_target")),
    ("Clearance is for the red cube alone -- bring it in.", ("red", "avoid_objects", "unspecified", "collect_target")),
    ("The yellow object is the single item you're allowed to move.", ("yellow", "avoid_objects", "unspecified", "collect_target")),
    ("Only the black block may be picked up.", ("black", "avoid_objects", "unspecified", "collect_target")),
    ("Permission extends to the blue target and no further.", ("blue", "avoid_objects", "unspecified", "collect_target")),
    ("The red die is the one object cleared for contact.", ("red", "avoid_objects", "unspecified", "collect_target")),
    ("You're authorised for the yellow cube only.", ("yellow", "avoid_objects", "unspecified", "collect_target")),
    ("Black object alone is yours to take.", ("black", "avoid_objects", "unspecified", "collect_target")),
    ("Contact is limited to the blue block.", ("blue", "avoid_objects", "unspecified", "collect_target")),
    ("The red target is the only piece you may lift.", ("red", "avoid_objects", "unspecified", "collect_target")),
    ("Just the yellow die is in scope for handling.", ("yellow", "avoid_objects", "unspecified", "collect_target")),
    ("Restrict contact to the black cube.", ("black", "avoid_objects", "unspecified", "collect_target")),

    # --- directional distractor: a ROUTE direction precedes the target's own location.
    # The route cardinal never shares a compass component with the target corner.
    ("Run the western edge on the way out, the blue die is in the northeast corner.", ("blue", "none", "NE", "collect_target")),
    ("Keep to the north side as you go, then collect the red cube from the southwest.", ("red", "none", "SW", "collect_target")),
    ("Move along the eastern boundary first; the yellow object is in the northwest.", ("yellow", "none", "NW", "collect_target")),
    ("Track the southern edge, then pick up the black block in the northeast.", ("black", "none", "NE", "collect_target")),
    ("Approach up the western flank -- the blue target sits in the southeast.", ("blue", "none", "SE", "collect_target")),
    ("Take the northern route out and grab the red die from the southwest corner.", ("red", "none", "SW", "collect_target")),
    ("Follow the eastern side down, then collect the yellow cube in the northwest.", ("yellow", "none", "NW", "collect_target")),
    ("Cut across the southern half before retrieving the black object in the northeast.", ("black", "none", "NE", "collect_target")),
    ("Stay along the western boundary, the blue block is in the southeast quadrant.", ("blue", "none", "SE", "collect_target")),
    ("Head up the northern edge; the red target is in the southwest.", ("red", "none", "SW", "collect_target")),
    ("Work the eastern margin, then take the yellow die from the northwest corner.", ("yellow", "none", "NW", "collect_target")),
    ("Come in along the southern side and collect the black cube in the northeast.", ("black", "none", "NE", "collect_target")),

    # --- avoid_objects carried by proximity/route language, no "avoid"-family verb ---
    ("Give the other dice plenty of room while you collect the blue block.", ("blue", "avoid_objects", "unspecified", "collect_target")),
    ("Keep a gap from the rest of the objects and bring the red cube in.", ("red", "avoid_objects", "unspecified", "collect_target")),
    ("Swing well wide of the other pieces to reach the yellow die.", ("yellow", "avoid_objects", "unspecified", "collect_target")),
    ("Leave space around the other objects on your way to the black target.", ("black", "avoid_objects", "unspecified", "collect_target")),
    ("Don't crowd the other dice -- collect the blue object and return.", ("blue", "avoid_objects", "unspecified", "collect_target")),
    ("Stay off the other pieces while you pull the red block out.", ("red", "avoid_objects", "unspecified", "collect_target")),
    ("Take a wide line past the other objects and grab the yellow cube.", ("yellow", "avoid_objects", "unspecified", "collect_target")),
    ("Keep separation from the rest of the field and bring the black die home.", ("black", "avoid_objects", "unspecified", "collect_target")),

    # --- the mirror case: GENERIC obstruction language carries NO constraint.
    # Follows eval_set's precedent ("navigate around any obstacles" -> none): only a
    # definite reference to the field's other objects sets avoid_objects.
    ("Collect the blue die, working around whatever is in your way.", ("blue", "none", "unspecified", "collect_target")),
    ("Get the red cube and pick a path through the clutter.", ("red", "none", "unspecified", "collect_target")),
    ("Bring the yellow object in, steering past any obstruction you meet.", ("yellow", "none", "unspecified", "collect_target")),
    ("Retrieve the black block, negotiating any obstacles en route.", ("black", "none", "unspecified", "collect_target")),
    ("Fetch the blue target and handle whatever terrain you hit.", ("blue", "none", "unspecified", "collect_target")),
    ("Go for the red die, routing past anything blocking you.", ("red", "none", "unspecified", "collect_target")),
    ("Collect the yellow cube and deal with obstructions as they come.", ("yellow", "none", "unspecified", "collect_target")),
    ("Bring the black object back, navigating whatever stands in the path.", ("black", "none", "unspecified", "collect_target")),

    # --- avoid_objects on a read_chip run ---
    ("Scan the blue die's chip and keep off the other objects.", ("blue", "avoid_objects", "unspecified", "read_chip")),
    ("Tag read on the red cube -- no contact with the rest.", ("red", "avoid_objects", "unspecified", "read_chip")),
    ("Get the chip code from the yellow block without touching the others.", ("yellow", "avoid_objects", "unspecified", "read_chip")),
    ("Read the black object's tag; the other dice stay untouched.", ("black", "avoid_objects", "unspecified", "read_chip")),
]

# Syntactic shapes absent from both training sets: neither model has an advantage,
# so this stratum measures raw robustness rather than template recall.
NOVEL = [
    # --- questions / requests ---
    ("Can you bring the blue die back for me?", ("blue", "none", "unspecified", "collect_target")),
    ("Could you read the tag on the red cube?", ("red", "none", "unspecified", "read_chip")),
    ("Would you head back to the start now?", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Mind grabbing the yellow object from the northeast?", ("yellow", "none", "NE", "collect_target")),
    ("Can we get the black block in without touching the others?", ("black", "avoid_objects", "unspecified", "collect_target")),
    ("Can you keep clear of the southwest corner?", ("unspecified", "avoid_regions", "SW", "collect_target")),

    # --- passive voice ---
    ("The blue target is to be collected and returned.", ("blue", "none", "unspecified", "collect_target")),
    ("The red die's chip is to be scanned.", ("red", "none", "unspecified", "read_chip")),
    ("The northeast quadrant is to be avoided.", ("unspecified", "avoid_regions", "NE", "collect_target")),
    ("All other objects are to be left untouched while the yellow cube is recovered.", ("yellow", "avoid_objects", "unspecified", "collect_target")),
    ("The run is to be aborted immediately.", ("unspecified", "none", "unspecified", "abort")),
    ("The vehicle is to be returned to the start.", ("unspecified", "none", "unspecified", "return_to_start")),

    # --- terse fragments ---
    ("Blue. Northwest. Go.", ("blue", "none", "NW", "collect_target")),
    ("Red cube. Chip only.", ("red", "none", "unspecified", "read_chip")),
    ("Abort.", ("unspecified", "none", "unspecified", "abort")),
    ("Home. Now.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Yellow, southeast, no contact with the others.", ("yellow", "avoid_objects", "SE", "collect_target")),
    ("Southwest closed.", ("unspecified", "avoid_regions", "SW", "collect_target")),

    # --- subordinate clauses ---
    ("If the blue die is still in the northeast, bring it in.", ("blue", "none", "NE", "collect_target")),
    ("Once you're clear of the start, collect the red object and return.", ("red", "none", "unspecified", "collect_target")),
    ("After the reset, scan the yellow cube's tag.", ("yellow", "none", "unspecified", "read_chip")),
    ("Since the northwest is blocked, plan around it.", ("unspecified", "avoid_regions", "NW", "collect_target")),
    ("Because the others must stay put, take only the black block.", ("black", "avoid_objects", "unspecified", "collect_target")),
    ("Whatever else you see, the black die is the only thing coming back.", ("black", "avoid_objects", "unspecified", "collect_target")),

    # --- indirect / polite ---
    ("I'd like the blue object brought back, please.", ("blue", "none", "unspecified", "collect_target")),
    ("We'd rather you didn't enter the southeast quarter.", ("unspecified", "avoid_regions", "SE", "collect_target")),
    ("Let's have the red die's chip read.", ("red", "none", "unspecified", "read_chip")),
    ("Best you come back to start now.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("I need you to stop everything and stay hidden.", ("unspecified", "none", "unspecified", "abort")),
    ("Please bring the yellow cube in without disturbing the rest.", ("yellow", "avoid_objects", "unspecified", "collect_target")),
]

# Second clean slice, written DURING the v6 training run and before any v6 result existed,
# because heldout_set.jsonl is now spent: round 4's templates were designed against its
# failures, so it measures recall for v6 the way eval_set.jsonl already did for v5. Weighted
# toward the classes round 4 changed (avoid_regions, avoid_objects, abort, read_chip), since
# that is where a rebalance can help or backfire.
HELDOUT2 = [
    # --- avoid_regions, wording distinct from round 4's templates ---
    ("Forget about the northwest for now.", ("unspecified", "avoid_regions", "NW", "collect_target")),
    ("The southeast is a dead zone this run.", ("unspecified", "avoid_regions", "SE", "collect_target")),
    ("Nobody goes into the northeast.", ("unspecified", "avoid_regions", "NE", "collect_target")),
    ("Southwest is a write-off, plan without it.", ("unspecified", "avoid_regions", "SW", "collect_target")),
    ("The northwest has been ruled out.", ("unspecified", "avoid_regions", "NW", "collect_target")),
    ("Pretend the southeast quarter does not exist.", ("unspecified", "avoid_regions", "SE", "collect_target")),
    ("Northeast is off your map now.", ("unspecified", "avoid_regions", "NE", "collect_target")),
    ("The southwest section has been struck.", ("unspecified", "avoid_regions", "SW", "collect_target")),
    ("They pulled the northwest from the field.", ("unspecified", "avoid_regions", "NW", "collect_target")),
    ("Southeast quadrant: hands off the whole area.", ("unspecified", "avoid_regions", "SE", "collect_target")),
    ("The northeast is not yours to drive through.", ("unspecified", "avoid_regions", "NE", "collect_target")),
    ("Southwest is barred to the vehicle.", ("unspecified", "avoid_regions", "SW", "collect_target")),
    ("Your route excludes the northwest entirely.", ("unspecified", "avoid_regions", "NW", "collect_target")),
    ("The southeast strip has been retired.", ("unspecified", "avoid_regions", "SE", "collect_target")),
    ("Northeast is a restricted band this round.", ("unspecified", "avoid_regions", "NE", "collect_target")),
    ("Nothing of yours crosses the southwest.", ("unspecified", "avoid_regions", "SW", "collect_target")),

    # --- avoid_objects, prohibition framing ---
    ("The blue die comes back; nothing near it gets moved.", ("blue", "avoid_objects", "unspecified", "collect_target")),
    ("Red cube only. The rest of the field is untouchable.", ("red", "avoid_objects", "unspecified", "collect_target")),
    ("Fetch the yellow block, and let the other dice be.", ("yellow", "avoid_objects", "unspecified", "collect_target")),
    ("Black target is the sole pickup, the others stay planted.", ("black", "avoid_objects", "unspecified", "collect_target")),
    ("Bring the blue object in without upsetting the arrangement.", ("blue", "avoid_objects", "unspecified", "collect_target")),
    ("Take the red die and let everything else sit.", ("red", "avoid_objects", "unspecified", "collect_target")),
    ("Yellow cube out, nothing else moves an inch.", ("yellow", "avoid_objects", "unspecified", "collect_target")),
    ("Collect the black block; the others are not yours.", ("black", "avoid_objects", "unspecified", "collect_target")),

    # --- avoid_objects, permission framing (still only half-learned by v5) ---
    ("The blue block is the extent of what you may touch.", ("blue", "avoid_objects", "unspecified", "collect_target")),
    ("Your remit covers the red die and nothing else.", ("red", "avoid_objects", "unspecified", "collect_target")),
    ("The yellow object is the whole of your clearance.", ("yellow", "avoid_objects", "unspecified", "collect_target")),
    ("You are cleared for the black cube, full stop.", ("black", "avoid_objects", "unspecified", "collect_target")),
    ("Handling rights: the blue die, and that is all.", ("blue", "avoid_objects", "unspecified", "collect_target")),
    ("The red target is where your permission ends.", ("red", "avoid_objects", "unspecified", "collect_target")),

    # --- avoid_objects, proximity framing, and on a read_chip run ---
    ("Leave a margin around the other dice and collect the yellow block.", ("yellow", "avoid_objects", "unspecified", "collect_target")),
    ("Keep off the rest of the field while you take the black object.", ("black", "avoid_objects", "unspecified", "collect_target")),
    ("Tag the blue cube and let the others alone.", ("blue", "avoid_objects", "unspecified", "read_chip")),
    ("Chip read on the red block, the others stay put.", ("red", "avoid_objects", "unspecified", "read_chip")),

    # --- plain retrieve ---
    ("Blue die, bring it in.", ("blue", "none", "unspecified", "collect_target")),
    ("The red object needs to come home.", ("red", "none", "unspecified", "collect_target")),
    ("Fetch the yellow block.", ("yellow", "none", "unspecified", "collect_target")),
    ("Black cube is the target.", ("black", "none", "unspecified", "collect_target")),
    ("Go and collect the blue target from the northeast.", ("blue", "none", "NE", "collect_target")),
    ("Red die is in the southwest.", ("red", "none", "SW", "collect_target")),
    ("The yellow object sits in the northwest.", ("yellow", "none", "NW", "collect_target")),
    ("Black block, southeast side.", ("black", "none", "SE", "collect_target")),
    ("The blue target has moved to the southwest.", ("blue", "none", "SW", "update_only")),
    ("Red object is now in the northeast.", ("red", "none", "NE", "update_only")),
    ("Yellow die shifted to the northwest.", ("yellow", "none", "NW", "update_only")),
    ("Black cube is now over in the southeast.", ("black", "none", "SE", "update_only")),

    # --- generic obstruction: the mirror case, still no constraint ---
    ("Collect the blue block, getting past whatever is in the way.", ("blue", "none", "unspecified", "collect_target")),
    ("Bring the red die in, around any junk on the floor.", ("red", "none", "unspecified", "collect_target")),
    ("Fetch the yellow object and cope with the terrain.", ("yellow", "none", "unspecified", "collect_target")),
    ("Get the black cube back, whatever the route throws up.", ("black", "none", "unspecified", "collect_target")),

    # --- directional distractor ---
    ("Run the eastern line out, then take the blue die from the northwest.", ("blue", "none", "NW", "collect_target")),
    ("Hold the southern edge, the red cube is in the northeast.", ("red", "none", "NE", "collect_target")),
    ("Come up the western side and collect the yellow object in the southeast.", ("yellow", "none", "SE", "collect_target")),
    ("Track the northern boundary, then grab the black block in the southwest.", ("black", "none", "SW", "collect_target")),

    # --- read_chip ---
    ("Tag on the blue die, read it.", ("blue", "none", "unspecified", "read_chip")),
    ("Chip number from the red cube, please.", ("red", "none", "unspecified", "read_chip")),
    ("Scan the yellow block and stay where you are.", ("yellow", "none", "unspecified", "read_chip")),
    ("Black object's tag -- read and report.", ("black", "none", "unspecified", "read_chip")),
    ("Read the blue target's chip in the southwest.", ("blue", "none", "SW", "read_chip")),
    ("Tag scan, red die, northeast.", ("red", "none", "NE", "read_chip")),
    ("Yellow cube in the northwest -- chip only.", ("yellow", "none", "NW", "read_chip")),
    ("Black block, southeast, read the tag.", ("black", "none", "SE", "read_chip")),
    ("We want the code off the blue object.", ("blue", "none", "unspecified", "read_chip")),
    ("Chip data from the red target, nothing more.", ("red", "none", "unspecified", "read_chip")),

    # --- abort ---
    ("Down tools. Stay where you are.", ("unspecified", "none", "unspecified", "abort")),
    ("That is enough -- stop and keep hidden.", ("unspecified", "none", "unspecified", "abort")),
    ("We are burned. Halt everything.", ("unspecified", "none", "unspecified", "abort")),
    ("No further action. Hold and stay covered.", ("unspecified", "none", "unspecified", "abort")),
    ("Stop the vehicle, the run is void.", ("unspecified", "none", "unspecified", "abort")),
    ("Hold it right there, stay unseen.", ("unspecified", "none", "unspecified", "abort")),
    ("Everything ceases now. Remain concealed.", ("unspecified", "none", "unspecified", "abort")),
    ("We are aborting -- no movement at all.", ("unspecified", "none", "unspecified", "abort")),

    # --- return_to_start ---
    ("Just bring it back, nothing collected.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Come in to the start, empty.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Send the vehicle home, we are done.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Back to the launch point, no cargo.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Bring the rover in and stop.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Return to the start zone, leave it all.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Head home, the pickup is off.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Get it back to base, nothing retrieved.", ("unspecified", "none", "unspecified", "return_to_start")),
]



# Third clean slice, written BEFORE the round-6 training run. The existing three sets contain
# zero examples of the six new actions, so they cannot measure them at all -- and heldout/
# heldout2 are at ceiling (1 error in 282) and can no longer detect a change either way.
# Weighted toward the new classes and, above all, toward the two boundaries most likely to
# blur: pause vs abort, and update_only vs collect_target.
HELDOUT3 = [
    # --- update_only: a statement correcting position on a running task ---
    ("The blue die has ended up in the northeast instead.", ("blue", "none", "NE", "update_only")),
    ("Amend the target position -- red cube, southwest.", ("red", "none", "SW", "update_only")),
    ("Note the yellow object is actually in the northwest.", ("yellow", "none", "NW", "update_only")),
    ("Black block's location was wrong; it's southeast.", ("black", "none", "SE", "update_only")),
    ("Revise to the northeast quadrant, same job.", ("unspecified", "none", "NE", "update_only")),
    ("For your information, the blue target drifted southwest.", ("blue", "none", "SW", "update_only")),
    ("Red object sits in the northwest now, keep at it.", ("red", "none", "NW", "update_only")),
    ("New info only: yellow die, southeast corner.", ("yellow", "none", "SE", "update_only")),
    ("The black cube turned up in the northeast after all.", ("black", "none", "NE", "update_only")),
    ("Positional correction -- blue, northwest.", ("blue", "none", "NW", "update_only")),

    # --- the update_only / collect_target boundary, in matched pairs ---
    ("The red target is in the southeast now instead.", ("red", "none", "SE", "update_only")),
    ("Fetch the red target from the southeast.", ("red", "none", "SE", "collect_target")),
    ("Yellow object, northwest corner, just so you know.", ("yellow", "none", "NW", "update_only")),
    ("Yellow object, northwest corner -- go and get it.", ("yellow", "none", "NW", "collect_target")),
    ("Black die has moved to the southwest.", ("black", "none", "SW", "update_only")),
    ("Collect the black die from the southwest.", ("black", "none", "SW", "collect_target")),

    # --- report_status ---
    ("Give me a rundown of what you're doing.", ("unspecified", "none", "unspecified", "report_status")),
    ("What's the situation out there?", ("unspecified", "none", "unspecified", "report_status")),
    ("Tell me your current task.", ("unspecified", "none", "unspecified", "report_status")),
    ("How far along are you?", ("unspecified", "none", "unspecified", "report_status")),
    ("I'd like a status check.", ("unspecified", "none", "unspecified", "report_status")),
    ("Report in.", ("unspecified", "none", "unspecified", "report_status")),
    ("What's on your plate right now?", ("unspecified", "none", "unspecified", "report_status")),
    ("Run me through your current orders.", ("unspecified", "none", "unspecified", "report_status")),
    ("Anything to report?", ("unspecified", "none", "unspecified", "report_status")),
    ("Confirm your understanding of the mission.", ("unspecified", "none", "unspecified", "report_status")),

    # --- pause: temporary, resumption expected ---
    ("Hang on a moment.", ("unspecified", "none", "unspecified", "pause")),
    ("Halt briefly, I'll get back to you.", ("unspecified", "none", "unspecified", "pause")),
    ("Just hold there for now.", ("unspecified", "none", "unspecified", "pause")),
    ("Put it on hold a second.", ("unspecified", "none", "unspecified", "pause")),
    ("Stay put briefly, more to come.", ("unspecified", "none", "unspecified", "pause")),
    ("Give it a rest for a minute.", ("unspecified", "none", "unspecified", "pause")),
    ("Interrupt the task, I'm not finished with you.", ("unspecified", "none", "unspecified", "pause")),
    ("Wait right there a moment.", ("unspecified", "none", "unspecified", "pause")),

    # --- the pause / abort boundary, in matched pairs on near-identical openings ---
    ("Stop there, I'll come back to you.", ("unspecified", "none", "unspecified", "pause")),
    ("Stop there, we're shutting it down.", ("unspecified", "none", "unspecified", "abort")),
    ("Hold where you are, back shortly.", ("unspecified", "none", "unspecified", "pause")),
    ("Hold where you are and keep out of sight.", ("unspecified", "none", "unspecified", "abort")),
    ("Everything stops for a moment.", ("unspecified", "none", "unspecified", "pause")),
    ("Everything stops, the attempt is void.", ("unspecified", "none", "unspecified", "abort")),

    # --- resume ---
    ("Right, press on.", ("unspecified", "none", "unspecified", "resume")),
    ("Pick the task back up.", ("unspecified", "none", "unspecified", "resume")),
    ("Continue where you stopped.", ("unspecified", "none", "unspecified", "resume")),
    ("You may proceed now.", ("unspecified", "none", "unspecified", "resume")),
    ("Crack on.", ("unspecified", "none", "unspecified", "resume")),
    ("Resume what you were doing.", ("unspecified", "none", "unspecified", "resume")),
    ("Onwards -- same objective.", ("unspecified", "none", "unspecified", "resume")),
    ("Start moving again.", ("unspecified", "none", "unspecified", "resume")),

    # --- retry_read ---
    ("That read didn't work, do it over.", ("unspecified", "none", "unspecified", "retry_read")),
    ("Attempt the tag scan again.", ("unspecified", "none", "unspecified", "retry_read")),
    ("Once more on the chip, please.", ("unspecified", "none", "unspecified", "retry_read")),
    ("I need that chip read done a second time.", ("unspecified", "none", "unspecified", "retry_read")),
    ("The scan came back empty -- go again.", ("unspecified", "none", "unspecified", "retry_read")),
    ("Another crack at reading the tag.", ("unspecified", "none", "unspecified", "retry_read")),

    # --- retry_send ---
    ("That didn't come through, transmit it again.", ("unspecified", "none", "unspecified", "retry_send")),
    ("Push the message out once more.", ("unspecified", "none", "unspecified", "retry_send")),
    ("Resend what you just sent.", ("unspecified", "none", "unspecified", "retry_send")),
    ("Another attempt at the transmission.", ("unspecified", "none", "unspecified", "retry_send")),
    ("Upload the code again.", ("unspecified", "none", "unspecified", "retry_send")),
    ("Send that through a second time.", ("unspecified", "none", "unspecified", "retry_send")),

    # --- the read_chip / collect_target boundary, the one error left after round 5 ---
    ("Tag only on the blue die -- leave it where it is.", ("blue", "none", "unspecified", "read_chip")),
    ("Nothing but the chip data from the red cube.", ("red", "none", "unspecified", "read_chip")),
    ("Yellow object: chip, not the object itself.", ("yellow", "none", "unspecified", "read_chip")),
    ("We want the black die's code and nothing more.", ("black", "none", "unspecified", "read_chip")),
    ("Just the tag from the blue block this time.", ("blue", "none", "unspecified", "read_chip")),
    ("Read-only on the red target.", ("red", "none", "unspecified", "read_chip")),

    # --- collect_target, to confirm the rename did not disturb the core verb ---
    ("Go and bring in the blue die.", ("blue", "none", "unspecified", "collect_target")),
    ("The red cube needs collecting from the northeast.", ("red", "none", "NE", "collect_target")),
    ("Pick up the yellow block in the southwest.", ("yellow", "none", "SW", "collect_target")),
    ("Retrieve the black object and come home.", ("black", "none", "unspecified", "collect_target")),
    ("Blue target, northwest -- fetch it.", ("blue", "none", "NW", "collect_target")),
    ("Bring the red die back with you.", ("red", "none", "unspecified", "collect_target")),

    # --- abort and return_to_start, to confirm pause/resume did not erode them ---
    ("Scrub it -- stop and stay concealed.", ("unspecified", "none", "unspecified", "abort")),
    ("The run is dead. Power down in place.", ("unspecified", "none", "unspecified", "abort")),
    ("Cease everything and stay unseen.", ("unspecified", "none", "unspecified", "abort")),
    ("Come back to the start, nothing collected.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Drive the vehicle home, we're finished.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Return empty to the launch point.", ("unspecified", "none", "unspecified", "return_to_start")),

    # --- avoid_regions, to confirm the new actions did not squeeze it ---
    ("The northwest is shut to you now.", ("unspecified", "avoid_regions", "NW", "collect_target")),
    ("Keep out of the southeast for the rest of this.", ("unspecified", "avoid_regions", "SE", "collect_target")),
    ("Northeast quarter is no longer available.", ("unspecified", "avoid_regions", "NE", "collect_target")),
    ("Nothing of yours goes into the southwest.", ("unspecified", "avoid_regions", "SW", "collect_target")),
    ("Treat the northwest as sealed off.", ("unspecified", "avoid_regions", "NW", "collect_target")),
    ("The southeast section is barred from here on.", ("unspecified", "avoid_regions", "SE", "collect_target")),
]

# Fourth clean slice, written BEFORE the round-7 run. heldout3 has now been used once to
# diagnose round 6, so it is no longer unspent. This set targets only the three boundaries
# that failed -- update_only/collect_target, read_chip/collect_target, pause/abort -- plus
# the read_chip/report_status collision the new actions introduced.
HELDOUT4 = [
    # --- update_only: varied markers, none of them the training wording ---
    ("Correction, the blue die is in the southeast.", ("blue", "none", "SE", "update_only")),
    ("Update -- red cube now northwest.", ("red", "none", "NW", "update_only")),
    ("The yellow object has relocated to the northeast.", ("yellow", "none", "NE", "update_only")),
    ("Ignore the earlier position; black block is southwest.", ("black", "none", "SW", "update_only")),
    ("Turns out the blue target is northeast after all.", ("blue", "none", "NE", "update_only")),
    ("Latest: red object, southwest quadrant.", ("red", "none", "SW", "update_only")),
    ("The yellow die isn't where we said -- northwest.", ("yellow", "none", "NW", "update_only")),
    ("Position revised for the black cube: southeast.", ("black", "none", "SE", "update_only")),

    # --- bare locative is a tasking, per the rule in schema.py ---
    ("The blue block is in the northeast corner.", ("blue", "none", "NE", "collect_target")),
    ("Red die sits in the southwest quadrant.", ("red", "none", "SW", "collect_target")),
    ("The yellow cube is over in the southeast.", ("yellow", "none", "SE", "collect_target")),
    ("Black object is up in the northwest area.", ("black", "none", "NW", "collect_target")),
    ("Blue target, northeast quarter.", ("blue", "none", "NE", "collect_target")),
    ("The red block is in the southwest.", ("red", "none", "SW", "collect_target")),

    # --- read_chip by exclusion, the failure that survived two rounds ---
    ("Chip alone from the blue die.", ("blue", "none", "unspecified", "read_chip")),
    ("Red cube -- tag, not the cube.", ("red", "none", "unspecified", "read_chip")),
    ("We only want the yellow object's number.", ("yellow", "none", "unspecified", "read_chip")),
    ("Black die: code and nothing further.", ("black", "none", "unspecified", "read_chip")),
    ("Solely a chip reading on the blue target.", ("blue", "none", "unspecified", "read_chip")),
    ("Read the red object, don't lift it.", ("red", "none", "unspecified", "read_chip")),
    ("The yellow die's tag is the entire task.", ("yellow", "none", "unspecified", "read_chip")),
    ("Scan the black cube and that's it.", ("black", "none", "unspecified", "read_chip")),

    # --- "only" on a collection, so it does not become a read_chip cue ---
    ("Just the blue die comes home with you.", ("blue", "none", "unspecified", "collect_target")),
    ("The red object alone needs bringing in.", ("red", "none", "unspecified", "collect_target")),
    ("Collect the yellow cube and nothing besides.", ("yellow", "none", "unspecified", "collect_target")),
    ("Only the black block is coming back.", ("black", "none", "unspecified", "collect_target")),

    # --- the "report" collision: object of the verb decides it ---
    ("Report the code from the blue die.", ("blue", "none", "unspecified", "read_chip")),
    ("Report how the run is going.", ("unspecified", "none", "unspecified", "report_status")),
    ("Send up the red target's tag number.", ("red", "none", "unspecified", "read_chip")),
    ("Send up a summary of your task.", ("unspecified", "none", "unspecified", "report_status")),
    ("What does the yellow object's tag say?", ("yellow", "none", "unspecified", "read_chip")),
    ("Give me a picture of where you're at.", ("unspecified", "none", "unspecified", "report_status")),

    # --- pause led by stop-language, which is how it failed in round 6 ---
    ("Stop a moment, I'll be back to you.", ("unspecified", "none", "unspecified", "pause")),
    ("Halt for a bit, not finished.", ("unspecified", "none", "unspecified", "pause")),
    ("Break for a second, more instructions coming.", ("unspecified", "none", "unspecified", "pause")),
    ("Come to a stop briefly.", ("unspecified", "none", "unspecified", "pause")),
    ("Quit what you're doing for now, stand by.", ("unspecified", "none", "unspecified", "pause")),
    ("Down tools briefly, we resume shortly.", ("unspecified", "none", "unspecified", "pause")),

    # --- abort with the same openings, differing only in finality ---
    ("Stop a moment -- actually, we're done entirely.", ("unspecified", "none", "unspecified", "abort")),
    ("Halt for good and stay concealed.", ("unspecified", "none", "unspecified", "abort")),
    ("Break off, the attempt is finished.", ("unspecified", "none", "unspecified", "abort")),
    ("Come to a stop and stay unseen.", ("unspecified", "none", "unspecified", "abort")),
    ("Quit what you're doing, run's over.", ("unspecified", "none", "unspecified", "abort")),
    ("Down tools, nothing further today.", ("unspecified", "none", "unspecified", "abort")),

    # --- resume and the retry pair, to confirm they hold ---
    ("Alright, back to it.", ("unspecified", "none", "unspecified", "resume")),
    ("Continue from where you paused.", ("unspecified", "none", "unspecified", "resume")),
    ("That read didn't land -- again please.", ("unspecified", "none", "unspecified", "retry_read")),
    ("Have another go at the chip.", ("unspecified", "none", "unspecified", "retry_read")),
    ("That send failed, go again.", ("unspecified", "none", "unspecified", "retry_send")),
    ("Transmit the code one more time.", ("unspecified", "none", "unspecified", "retry_send")),
]


# Sixth clean slice, and the ONLY instrument for the schema-v3 constraint fields. Written
# BEFORE v13 trains and never consulted while writing the round-10 training block, which is
# the discipline round 9 broke: it wrote training templates against observed heldout3/heldout4
# failures, which spends a set as an instrument because a gain there then partly measures how
# well the failures were paraphrased.
#
# Idioms here are deliberately disjoint from round 10's containment templates. Round 10 used
# stay within / keep the vehicle in / operate only in / confined to / hold to the far /
# remain inside / your working area / do not leave. None of those appear below. If the model
# only answers to the eight verbs it was trained on, these sentences expose it -- which is
# exactly the failure avoid_region showed on heldout5 (6 cue words, 0 for 4 on novel wording).
#
# label tuple = (target_color, constraints, target_location, action[, stay, until, avoid])
#
# STRATA, scored separately:
#   stay_half      - containment in halves, novel verbs
#   stay_quad      - containment in a quadrant, with a named target
#   pathway        - stay_region="pathway", the rules' undefined safe corridor
#   stay_until     - containment WITH a release condition (the rules' hardest shape)
#   avoid_stay     - both constraints in one command (the reason v3 split the field)
#   avoid_target   - avoid_region alongside a named target: v11's documented blind spot
#   route_negative - route verbs + region words, NO constraint. Over-firing detector; if
#                    these fail, the model learned "direction word => constraint".
HELDOUT6 = [
    # --- stay_half: containment, halves, none of round 10's verbs ---
    ("Don't stray out of the northern half at any stage.", ("unspecified", "none", "unspecified", "collect_target", "N"), "stay_half"),
    ("Your box for this attempt is the southern half.", ("unspecified", "none", "unspecified", "collect_target", "S"), "stay_half"),
    ("Everything happens in the eastern half, nothing outside it.", ("unspecified", "none", "unspecified", "collect_target", "E"), "stay_half"),
    ("Limit yourself to the western half please.", ("unspecified", "none", "unspecified", "collect_target", "W"), "stay_half"),
    ("You've got the north half and that's it.", ("unspecified", "none", "unspecified", "collect_target", "N"), "stay_half"),
    ("Bounded by the south half for the duration.", ("unspecified", "none", "unspecified", "collect_target", "S"), "stay_half"),
    ("Go no further west than the halfway line -- east half only.", ("unspecified", "none", "unspecified", "collect_target", "E"), "stay_half"),
    ("The whole run takes place in the west half of the course.", ("unspecified", "none", "unspecified", "collect_target", "W"), "stay_half"),
    ("Is the north half my limit? Yes, treat it that way.", ("unspecified", "none", "unspecified", "collect_target", "N"), "stay_half"),
    ("South half. Nowhere else.", ("unspecified", "none", "unspecified", "collect_target", "S"), "stay_half"),

    # --- stay_quad: containment in a quadrant, target named ---
    ("The yellow die and the northwest quadrant -- that's your whole world right now.", ("yellow", "none", "unspecified", "collect_target", "NW"), "stay_quad"),
    ("Don't wander out of the northeast corner; the red cube is in there.", ("red", "none", "unspecified", "collect_target", "NE"), "stay_quad"),
    ("Everything you need is in the southwest section, so don't go past it. Blue object.", ("blue", "none", "unspecified", "collect_target", "SW"), "stay_quad"),
    ("Black target, and your movement is capped at the southeast quadrant.", ("black", "none", "unspecified", "collect_target", "SE"), "stay_quad"),
    ("Boxed into the northwest corner for this one. Bring in the blue piece.", ("blue", "none", "unspecified", "collect_target", "NW"), "stay_quad"),
    ("Yellow is the pickup and the southeast section is as far as you go.", ("yellow", "none", "unspecified", "collect_target", "SE"), "stay_quad"),

    # --- pathway: the rules name a safe corridor but never define it geometrically ---
    ("Wheels on the marked path the whole time, and fetch the red block.", ("red", "none", "unspecified", "collect_target", "pathway"), "pathway"),
    ("You may not deviate from the safe corridor.", ("unspecified", "none", "unspecified", "collect_target", "pathway"), "pathway"),
    ("Blue object please, and no wandering off the taped route.", ("blue", "none", "unspecified", "collect_target", "pathway"), "pathway"),
    ("The corridor is the constraint -- don't step outside it.", ("unspecified", "none", "unspecified", "collect_target", "pathway"), "pathway"),
    ("Collect the black cube. Wheels never leave the marked track.", ("black", "none", "unspecified", "collect_target", "pathway"), "pathway"),
    ("Is it fine to shortcut off the path? No. Stick to it.", ("unspecified", "none", "unspecified", "collect_target", "pathway"), "pathway"),

    # --- stay_until: containment with a release condition ---
    ("You're held to the southern half right up until the east half, then it's open.", ("unspecified", "none", "unspecified", "collect_target", "S", "E"), "stay_until"),
    ("North half is the cap until you make the west side of the field.", ("unspecified", "none", "unspecified", "collect_target", "N", "W"), "stay_until"),
    ("Don't stray from the east half before you've got to the north half.", ("unspecified", "none", "unspecified", "collect_target", "E", "N"), "stay_until"),
    ("Bounded to the west half; reaching the south half clears it.", ("unspecified", "none", "unspecified", "collect_target", "W", "S"), "stay_until"),
    ("Down in the southern half until the eastern half, and the yellow die is the pickup.", ("yellow", "none", "unspecified", "collect_target", "S", "E"), "stay_until"),
    ("Your limit is the north half, lifted the moment you hit the west half.", ("unspecified", "none", "unspecified", "collect_target", "N", "W"), "stay_until"),
    ("East half only, and that holds until the north half.", ("unspecified", "none", "unspecified", "collect_target", "E", "N"), "stay_until"),
    ("Boxed to the west side of the field until the south half -- then grab the black block.", ("black", "none", "unspecified", "collect_target", "W", "S"), "stay_until"),
    ("When does the southern-half restriction end? When you reach the eastern half.", ("unspecified", "none", "unspecified", "collect_target", "S", "E"), "stay_until"),
    ("Held in the northern half. Release point is the western half.", ("unspecified", "none", "unspecified", "collect_target", "N", "W"), "stay_until"),

    # --- avoid_stay: both constraints in one command ---
    ("Northern half is your bound and the southwest corner is shut.", ("unspecified", "none", "unspecified", "collect_target", "N", "none", "SW"), "avoid_stay"),
    ("Don't stray from the south half, and the northeast quadrant is a no-go.", ("unspecified", "none", "unspecified", "collect_target", "S", "none", "NE"), "avoid_stay"),
    ("You're capped at the east half; also the northwest corner is out of play.", ("unspecified", "none", "unspecified", "collect_target", "E", "none", "NW"), "avoid_stay"),
    ("West half is the limit. Nothing goes into the southeast section.", ("unspecified", "none", "unspecified", "collect_target", "W", "none", "SE"), "avoid_stay"),
    ("Boxed into the north half with the southwest quadrant pulled from the map.", ("unspecified", "none", "unspecified", "collect_target", "N", "none", "SW"), "avoid_stay"),
    ("Your world is the southern half, minus the northeast corner.", ("unspecified", "none", "unspecified", "collect_target", "S", "none", "NE"), "avoid_stay"),
    ("Keep to the marked corridor and the northwest quarter is closed.", ("unspecified", "none", "unspecified", "collect_target", "pathway", "none", "NW"), "avoid_stay"),
    # The one row where two region slots share a compass letter (E and SE). Kept on
    # purpose: "east half" and "southeast corner" are different words, so it is hard, not
    # ambiguous, and collapsing the two is a failure worth being able to see. Every other
    # row in this file keeps its region slots compass-disjoint.
    ("East half, and scratch the southeast corner from your plan.", ("unspecified", "none", "unspecified", "collect_target", "E", "none", "SE"), "avoid_stay"),

    # --- avoid_target: a keep-out region alongside a NAMED target. v11 had 0 of 212 such
    #     rows and missed every heldout5 sentence of this shape. Round 10 added 32.
    ("Yellow cube is the objective, and the southwest corner has been pulled.", ("yellow", "avoid_regions", "SW", "collect_target"), "avoid_target"),
    ("Bring the black piece in. Northeast quadrant is sealed.", ("black", "avoid_regions", "NE", "collect_target"), "avoid_target"),
    ("We want the blue die; the northwest section is no longer on the map.", ("blue", "avoid_regions", "NW", "collect_target"), "avoid_target"),
    ("Red block, and whatever you do stay out of the southeast corner.", ("red", "avoid_regions", "SE", "collect_target"), "avoid_target"),
    ("The southern half has been fenced off -- the yellow object still comes home.", ("yellow", "avoid_regions", "S", "collect_target"), "avoid_target"),
    ("Grab the blue target. North half is unavailable this round.", ("blue", "avoid_regions", "N", "collect_target"), "avoid_target"),
    ("Can I cut through the east half? No, it's shut. Black cube is the pickup.", ("black", "avoid_regions", "E", "collect_target"), "avoid_target"),
    ("Red object is yours and the west half is scratched.", ("red", "avoid_regions", "W", "collect_target"), "avoid_target"),

    # --- route_negative: region words with ROUTE verbs and no constraint at all. The cue is
    #     the verb: "cut across the south half" is a path, "stay to the south half" is a
    #     fence. If these come back with a stay_region set, the model is firing on the
    #     direction word rather than the verb.
    ("Swing through the southern half and collect the blue die from the northeast corner.", ("blue", "none", "NE", "collect_target"), "route_negative"),
    ("Your line runs up the western half -- the red cube is in the northeast section.", ("red", "none", "NE", "collect_target"), "route_negative"),
    ("Come in over the north half, black object is in the southwest quadrant.", ("black", "none", "SW", "collect_target"), "route_negative"),
    ("Drive the east half on approach; yellow block sits in the northwest corner.", ("yellow", "none", "NW", "collect_target"), "route_negative"),
    ("Route takes you diagonally across the south half to the blue target in the northwest.", ("blue", "none", "NW", "collect_target"), "route_negative"),
    ("Head up the west half, then the red die in the southeast section is yours.", ("red", "none", "SE", "collect_target"), "route_negative"),
    ("Push across the northern half to reach the yellow object in the southeast.", ("yellow", "none", "SE", "collect_target"), "route_negative"),
    ("Via the eastern half, please -- black piece, southwest corner.", ("black", "none", "SW", "collect_target"), "route_negative"),
]

# Seventh clean slice, written BEFORE round 11's training data and BEFORE v14 trains.
#
# Round 10 is about to spend heldout6: the v13 error dump on that set is the input to round
# 11's templates, so a later gain on heldout6 partly measures how well those failures were
# paraphrased. heldout6 becomes a DEV set from here on and this file replaces it as the clean
# instrument. That is the same trade round 9 made by accident with heldout3/heldout4; making
# it deliberately, and writing the replacement first, is the whole difference.
#
# Honest limit on "clean": the guarantee is mechanical -- build() rejects any exact overlap
# and reports anything >=0.85 similar to train/eval/heldout1-6. It is NOT "the author had
# never seen this set", because the same author writes round 11 afterwards. Different idioms
# were chosen deliberately; the similarity check is what actually enforces it.
#
# Every stratum targets a v13 failure mode, in INDEPENDENT wording from both round 10's
# training templates and heldout6:
#   stay_novel     - containment, a third distinct idiom family (19 missed on heldout6)
#   pause_v_abort  - same opening verb, one modifier decides. 8 of v13's 23 action errors.
#   abort_v_return - "stop AND stay put" is abort, not return_to_start (3 errors)
#   resume_v_retry - "again" alone is resume; retry needs a read or a send (3 errors)
#   chip_terse     - "chip only" in fragments (2 errors)
#   avoid_halves   - keep-out stated as a half, not a quadrant (16 training rows only)
#   route_neg      - route verbs, no constraint. Over-firing control; v13 scored 8/8.
#
# label tuple = (target_color, constraints, target_location, action[, stay, until, avoid])
HELDOUT7 = [
    # --- stay_novel: containment verbs unlike round 10's and unlike heldout6's ---
    ("Keep it to the northern half, please.", ("unspecified", "none", "unspecified", "collect_target", "N"), "stay_novel"),
    ("The southern half is where you live this round.", ("unspecified", "none", "unspecified", "collect_target", "S"), "stay_novel"),
    ("No leaving the eastern half. At all.", ("unspecified", "none", "unspecified", "collect_target", "E"), "stay_novel"),
    ("You're penned into the western half.", ("unspecified", "none", "unspecified", "collect_target", "W"), "stay_novel"),
    ("Work exclusively in the north half of the course.", ("unspecified", "none", "unspecified", "collect_target", "N"), "stay_novel"),
    ("Anything outside the south half is out of bounds.", ("unspecified", "none", "unspecified", "collect_target", "S"), "stay_novel"),
    ("Restrict all movement to the east half.", ("unspecified", "none", "unspecified", "collect_target", "E"), "stay_novel"),
    ("The west half is your envelope for this attempt.", ("unspecified", "none", "unspecified", "collect_target", "W"), "stay_novel"),
    ("You may not exit the northern half.", ("unspecified", "none", "unspecified", "collect_target", "N"), "stay_novel"),
    ("Stick to the south half and bring in the blue die.", ("blue", "none", "unspecified", "collect_target", "S"), "stay_novel"),
    ("The northwest quadrant is the play area -- red cube is the pickup.", ("red", "none", "unspecified", "collect_target", "NW"), "stay_novel"),
    ("Treat the southeast corner boundary as hard. Yellow object.", ("yellow", "none", "unspecified", "collect_target", "SE"), "stay_novel"),
    ("Inside the marked lane, always. Black block please.", ("black", "none", "unspecified", "collect_target", "pathway"), "stay_novel"),
    ("The taped corridor contains you for the whole run.", ("unspecified", "none", "unspecified", "collect_target", "pathway"), "stay_novel"),
    ("Penned to the eastern half until you make the northern half.", ("unspecified", "none", "unspecified", "collect_target", "E", "N"), "stay_novel"),
    ("Keep it to the west half; the south half releases you.", ("unspecified", "none", "unspecified", "collect_target", "W", "S"), "stay_novel"),

    # --- pause_v_abort: identical opening, a single modifier decides. These are the pairs. ---
    ("Halt -- back shortly.", ("unspecified", "none", "unspecified", "pause"), "pause_v_abort"),
    ("Halt -- that's final.", ("unspecified", "none", "unspecified", "abort"), "pause_v_abort"),
    ("Standby mode, I'll call you.", ("unspecified", "none", "unspecified", "pause"), "pause_v_abort"),
    ("Standby permanently, we're out.", ("unspecified", "none", "unspecified", "abort"), "pause_v_abort"),
    ("Freeze for the moment.", ("unspecified", "none", "unspecified", "pause"), "pause_v_abort"),
    ("Freeze -- mission terminated.", ("unspecified", "none", "unspecified", "abort"), "pause_v_abort"),
    ("Suspend it, I'm not finished.", ("unspecified", "none", "unspecified", "pause"), "pause_v_abort"),
    ("Suspend it, we're finished.", ("unspecified", "none", "unspecified", "abort"), "pause_v_abort"),
    ("Hold fire, more coming.", ("unspecified", "none", "unspecified", "pause"), "pause_v_abort"),
    ("Hold fire and stay dark, we're done.", ("unspecified", "none", "unspecified", "abort"), "pause_v_abort"),
    ("Stall there a second.", ("unspecified", "none", "unspecified", "pause"), "pause_v_abort"),
    ("Stall there, the run is over.", ("unspecified", "none", "unspecified", "abort"), "pause_v_abort"),
    ("Downtime, brief.", ("unspecified", "none", "unspecified", "pause"), "pause_v_abort"),
    ("Downtime, permanent.", ("unspecified", "none", "unspecified", "abort"), "pause_v_abort"),

    # --- abort_v_return: both say "stop". The cue is whether the vehicle is told to STAY
    #     where it is (abort) or to come home (return_to_start). v13 read three aborts as
    #     returns despite explicit "hold where you sit" wording.
    ("Knock it off and don't budge.", ("unspecified", "none", "unspecified", "abort"), "abort_v_return"),
    ("Knock it off and come on home.", ("unspecified", "none", "unspecified", "return_to_start"), "abort_v_return"),
    ("That's enough -- park it where it stands.", ("unspecified", "none", "unspecified", "abort"), "abort_v_return"),
    ("That's enough -- drive it back to the line.", ("unspecified", "none", "unspecified", "return_to_start"), "abort_v_return"),
    ("Wrap up and stay exactly there, out of sight.", ("unspecified", "none", "unspecified", "abort"), "abort_v_return"),
    ("Wrap up and bring it in empty.", ("unspecified", "none", "unspecified", "return_to_start"), "abort_v_return"),
    ("Jack it in. No movement, stay concealed.", ("unspecified", "none", "unspecified", "abort"), "abort_v_return"),
    ("Jack it in and return to the starting square.", ("unspecified", "none", "unspecified", "return_to_start"), "abort_v_return"),

    # --- resume_v_retry: "again" is the trap. Resuming a paused run is not retrying an
    #     operation; a retry names the thing being redone (the scan, the transmission).
    ("Away you go, pick up where you left off.", ("unspecified", "none", "unspecified", "resume"), "resume_v_retry"),
    ("Put the reader back on that tag.", ("unspecified", "none", "unspecified", "retry_read"), "resume_v_retry"),
    ("Push the code out again.", ("unspecified", "none", "unspecified", "retry_send"), "resume_v_retry"),
    ("Back underway, carry on.", ("unspecified", "none", "unspecified", "resume"), "resume_v_retry"),
    ("Fire it up once more and keep going.", ("unspecified", "none", "unspecified", "resume"), "resume_v_retry"),
    ("The tag didn't take -- one more pass on it.", ("unspecified", "none", "unspecified", "retry_read"), "resume_v_retry"),
    ("Transmission dropped, put it through once more.", ("unspecified", "none", "unspecified", "retry_send"), "resume_v_retry"),
    ("You're clear to move again.", ("unspecified", "none", "unspecified", "resume"), "resume_v_retry"),

    # --- chip_terse: read_chip stated as a fragment, the shape v13 read as collect_target
    ("Blue block. Tag data only.", ("blue", "none", "unspecified", "read_chip"), "chip_terse"),
    ("Red die, southeast -- data, not the object.", ("red", "none", "SE", "read_chip"), "chip_terse"),
    ("Yellow object: scan, don't lift.", ("yellow", "none", "unspecified", "read_chip"), "chip_terse"),
    ("Black cube. Read it, leave it.", ("black", "none", "unspecified", "read_chip"), "chip_terse"),
    ("Just the numbers off the blue target.", ("blue", "none", "unspecified", "read_chip"), "chip_terse"),
    ("Red piece in the northwest. Scan only.", ("red", "none", "NW", "read_chip"), "chip_terse"),

    # --- avoid_halves: a keep-out region stated as a HALF. Training had 16 such rows and
    #     v13 missed "fenced off" and "is scratched" on both.
    ("The northern half is walled off for this attempt.", ("unspecified", "avoid_regions", "N", "collect_target"), "avoid_halves"),
    ("Write off the south half, it's not in play.", ("unspecified", "avoid_regions", "S", "collect_target"), "avoid_halves"),
    ("Keep every wheel out of the east half.", ("unspecified", "avoid_regions", "E", "collect_target"), "avoid_halves"),
    ("The west half has been struck from the course.", ("unspecified", "avoid_regions", "W", "collect_target"), "avoid_halves"),
    ("Blue die is the pickup; the northern half is barred.", ("blue", "avoid_regions", "N", "collect_target"), "avoid_halves"),
    ("Grab the red cube. South half: forbidden.", ("red", "avoid_regions", "S", "collect_target"), "avoid_halves"),
    ("Yellow object comes home and the east half stays empty of you.", ("yellow", "avoid_regions", "E", "collect_target"), "avoid_halves"),
    ("We've closed the west half. Black block is still yours.", ("black", "avoid_regions", "W", "collect_target"), "avoid_halves"),

    # --- route_neg: region words under route verbs, no constraint. Control stratum: v13
    #     scored 8/8 here and round 11 adds containment breadth, which is exactly the change
    #     most likely to break it by making the model trigger-happy.
    ("Run the northern half on your way to the blue cube in the southeast.", ("blue", "none", "SE", "collect_target"), "route_neg"),
    ("Approach over the south half; red object is in the northwest corner.", ("red", "none", "NW", "collect_target"), "route_neg"),
    ("Your path cuts the east half -- yellow die is in the southwest.", ("yellow", "none", "SW", "collect_target"), "route_neg"),
    ("Come through the west half to the black block in the northeast.", ("black", "none", "NE", "collect_target"), "route_neg"),
    ("Transit the northern half, then read the chip on the blue target.", ("blue", "none", "unspecified", "read_chip"), "route_neg"),
    ("Out across the south half and back, nothing to collect.", ("unspecified", "none", "unspecified", "return_to_start"), "route_neg"),
]

# Eighth clean slice, written BEFORE round 12's templates.
#
# heldout7 is spent: I read v14's error dump on it to diagnose the avoid<->stay polarity
# confusion, so a round-12 gain there partly measures how well those failures were
# paraphrased. Same bookkeeping as heldout6 before it. The treadmill is real and is called
# out in the commit: every round costs an instrument, and the way off it is test data written
# by someone who is not the author of the training set.
#
# COMPLEMENT CONSTRUCTIONS are the point of this set. v14's polarity errors were not random
# -- they clustered on sentences that state a constraint by naming what is EXCLUDED rather
# than what is permitted ("anything outside the south half is out of bounds" means stay=S,
# not avoid=S). Each polarity pair below is the same words with the quantifier flipped, so
# the only cue is scope, and a model that keys on vocabulary alone must fail one side.
#
# label tuple = (target_color, constraints, target_location, action[, stay, until, avoid])
HELDOUT8 = [
    # --- polarity_pair: identical frame, "everywhere except X" (stay) vs "X" (avoid) ---
    ("Everywhere except the northern half is off the table.", ("unspecified", "none", "unspecified", "collect_target", "N"), "polarity_pair"),
    ("The northern half is off the table.", ("unspecified", "avoid_regions", "N", "collect_target"), "polarity_pair"),
    ("Anywhere other than the south half is a problem.", ("unspecified", "none", "unspecified", "collect_target", "S"), "polarity_pair"),
    ("The south half is a problem -- route around it.", ("unspecified", "avoid_regions", "S", "collect_target"), "polarity_pair"),
    ("All but the eastern half is denied to you.", ("unspecified", "none", "unspecified", "collect_target", "E"), "polarity_pair"),
    ("The eastern half is denied to you.", ("unspecified", "avoid_regions", "E", "collect_target"), "polarity_pair"),
    ("Everything but the west half is a lost cause.", ("unspecified", "none", "unspecified", "collect_target", "W"), "polarity_pair"),
    ("The west half is a lost cause -- keep off it.", ("unspecified", "avoid_regions", "W", "collect_target"), "polarity_pair"),
    ("Nothing beyond the northwest corner is permitted.", ("unspecified", "none", "unspecified", "collect_target", "NW"), "polarity_pair"),
    ("Nothing within the northwest corner is permitted.", ("unspecified", "avoid_regions", "NW", "collect_target"), "polarity_pair"),
    ("Leaving the southeast section is the violation.", ("unspecified", "none", "unspecified", "collect_target", "SE"), "polarity_pair"),
    ("Entering the southeast section is the violation.", ("unspecified", "avoid_regions", "SE", "collect_target"), "polarity_pair"),

    # --- polarity_single: one side only, mixed, with targets named ---
    ("You're confined to the north half. Blue die.", ("blue", "none", "unspecified", "collect_target", "N"), "polarity_single"),
    ("The north half is a dead zone. Blue die is still the pickup.", ("blue", "avoid_regions", "N", "collect_target"), "polarity_single"),
    ("Red cube, and you're not to exit the southern half.", ("red", "none", "unspecified", "collect_target", "S"), "polarity_single"),
    ("Red cube. You're not to enter the southern half.", ("red", "avoid_regions", "S", "collect_target"), "polarity_single"),
    ("Yellow block is yours; the marked corridor is where you live.", ("yellow", "none", "unspecified", "collect_target", "pathway"), "polarity_single"),
    ("Black object, and the northeast quarter is a write-off.", ("black", "avoid_regions", "NE", "collect_target"), "polarity_single"),
    ("Restricted to the southwest corner. Yellow target.", ("yellow", "none", "unspecified", "collect_target", "SW"), "polarity_single"),
    ("The southwest corner is restricted. Yellow target.", ("yellow", "avoid_regions", "SW", "collect_target"), "polarity_single"),

    # --- stop_verbs: abort vs pause vs return_to_start. 15 of v14's 19 action errors are in
    #     this cluster. Same opening, the modifier or the movement cue decides.
    ("Cut the power -- briefly.", ("unspecified", "none", "unspecified", "pause"), "stop_verbs"),
    ("Cut the power, we're finished.", ("unspecified", "none", "unspecified", "abort"), "stop_verbs"),
    ("Down you go, back shortly.", ("unspecified", "none", "unspecified", "pause"), "stop_verbs"),
    ("Down you go for good, stay out of sight.", ("unspecified", "none", "unspecified", "abort"), "stop_verbs"),
    ("Drop everything and sit there a moment.", ("unspecified", "none", "unspecified", "pause"), "stop_verbs"),
    ("Drop everything and sit there -- that's the run.", ("unspecified", "none", "unspecified", "abort"), "stop_verbs"),
    ("Drop everything and bring yourself back.", ("unspecified", "none", "unspecified", "return_to_start"), "stop_verbs"),
    ("Enough. Freeze and keep hidden.", ("unspecified", "none", "unspecified", "abort"), "stop_verbs"),
    ("Enough. Trundle on back to the line.", ("unspecified", "none", "unspecified", "return_to_start"), "stop_verbs"),
    ("Quit and hold, nobody should see you.", ("unspecified", "none", "unspecified", "abort"), "stop_verbs"),
    ("Quit and make for the start.", ("unspecified", "none", "unspecified", "return_to_start"), "stop_verbs"),
    ("Simmer down a second, I'll call it.", ("unspecified", "none", "unspecified", "pause"), "stop_verbs"),
    ("Simmer down, the attempt is void.", ("unspecified", "none", "unspecified", "abort"), "stop_verbs"),
    ("Go dark and don't move. We're out.", ("unspecified", "none", "unspecified", "abort"), "stop_verbs"),
    ("Empty-handed, head for base.", ("unspecified", "none", "unspecified", "return_to_start"), "stop_verbs"),
    ("Hold it -- I'll tell you when.", ("unspecified", "none", "unspecified", "pause"), "stop_verbs"),

    # --- resume_retry: "again" without an operation is resume; retry names the scan or send ---
    ("Wheels turning, please.", ("unspecified", "none", "unspecified", "resume"), "resume_retry"),
    ("Take it from where you stopped.", ("unspecified", "none", "unspecified", "resume"), "resume_retry"),
    ("You're released -- proceed.", ("unspecified", "none", "unspecified", "resume"), "resume_retry"),
    ("That chip never came through. Again.", ("unspecified", "none", "unspecified", "retry_read"), "resume_retry"),
    ("The tag needs rereading.", ("unspecified", "none", "unspecified", "retry_read"), "resume_retry"),
    ("Nothing arrived on our end -- push the code out.", ("unspecified", "none", "unspecified", "retry_send"), "resume_retry"),
    ("Broadcast those digits one more time.", ("unspecified", "none", "unspecified", "retry_send"), "resume_retry"),
    ("Resume, and no need to rescan anything.", ("unspecified", "none", "unspecified", "resume"), "resume_retry"),

    # --- controls: no constraint at all, and a plain read_chip. Over-firing detector: round 12
    #     adds a lot of polarity vocabulary, which is the change most likely to make the model
    #     see a constraint in a sentence that has none.
    ("Collect the blue block and come back.", ("blue", "none", "unspecified", "collect_target"), "controls"),
    ("Red die is in the northeast. Go.", ("red", "none", "NE", "collect_target"), "controls"),
    ("Read the chip on the yellow object.", ("yellow", "none", "unspecified", "read_chip"), "controls"),
    ("Black cube, southwest. Chip data, leave it be.", ("black", "none", "SW", "read_chip"), "controls"),
    ("Swing wide around the southern half on your way to the blue target in the northeast.", ("blue", "none", "NE", "collect_target"), "controls"),
    ("Your line runs the eastern half; the yellow die is in the northwest.", ("yellow", "none", "NW", "collect_target"), "controls"),
    ("What are you working on?", ("unspecified", "none", "unspecified", "report_status"), "controls"),
    ("Current task, please.", ("unspecified", "none", "unspecified", "report_status"), "controls"),
]

# Above this similarity to any existing train/eval sentence, a "fresh" sentence is a
# paraphrase and the held-out set stops being held out. Tuned so the genuine
# near-misses surface for inspection rather than silently passing.
NEAR_DUP_THRESHOLD = 0.85


def load_texts(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["text"] for line in f]


def build(tagged, out_path, extra_sources):
    seen = {}
    for text, _, tag in tagged:
        if text in seen:
            raise SystemExit(f"duplicate held-out text ({seen[text]} vs {tag}): {text!r}")
        seen[text] = tag

    existing = {"train": load_texts("data/train_set.jsonl"),
                "eval": load_texts("data/eval_set.jsonl")}
    for name, path in extra_sources.items():
        existing[name] = load_texts(path)

    # exact overlap is fatal -- it would make this a training set
    for src, texts in existing.items():
        overlap = sorted(set(seen) & set(texts))
        if overlap:
            raise SystemExit(f"{src} overlap, aborting write: {overlap}")

    # near-duplicates are reported, not silently accepted: a paraphrase of a training
    # sentence measures recall, not generalisation
    flagged = []
    for text in seen:
        for src, texts in existing.items():
            match = max(texts, key=lambda t: difflib.SequenceMatcher(None, text, t).ratio())
            ratio = difflib.SequenceMatcher(None, text, match).ratio()
            if ratio >= NEAR_DUP_THRESHOLD:
                flagged.append((ratio, src, text, match))
    if flagged:
        print(f"WARNING: {len(flagged)} sentence(s) at or above {NEAR_DUP_THRESHOLD} similarity:")
        for ratio, src, text, match in sorted(flagged, reverse=True):
            print(f"  {ratio:.3f} vs {src}\n    NEW: {text}\n    OLD: {match}")

    with open(out_path, "w", encoding="utf-8") as f:
        for text, labels, tag in tagged:
            # 4-tuples are the v2 authoring shape; schema.expand_v2 maps them to the
            # v3 fields (and rejects any value outside the schema enums). A 5th/6th
            # element carries stay_region/until_region for the rows that need them.
            row = {"text": text, "expected": expand_v2(*labels), "set": tag}
            f.write(json.dumps(row) + "\n")
    counts = collections.Counter(tag for _, _, tag in tagged)
    mix = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"wrote {len(tagged)} examples to {out_path}  [{mix}]  "
          f"0 exact overlap with {', '.join(sorted(existing))}")


# Fifth clean slice, written BEFORE the round-9 run and BEFORE v12 trains. Round 9 wrote
# training templates against observed heldout3/heldout4 failures, which spends both as
# instruments: a gain there now partly measures how well those templates were paraphrased.
# This set exists so v12 has one surface nobody has tuned against. Same boundaries, wording
# chosen to share no idiom with the round-9 pools -- no "down tools", "come to a stop",
# "crack on", "press on", "stop there", or mid-sentence "actually"/"correction".
HELDOUT5 = [
    # --- pause: temporary, expects to resume. Idioms absent from training ---
    ("Idle for a tick, I'm not done with you.", ("unspecified", "none", "unspecified", "pause")),
    ("Sit on your hands a minute.", ("unspecified", "none", "unspecified", "pause")),
    ("Nothing for now -- await my word.", ("unspecified", "none", "unspecified", "pause")),
    ("Ease off, I'll signal when to go.", ("unspecified", "none", "unspecified", "pause")),
    ("Put a pin in it, more to come.", ("unspecified", "none", "unspecified", "pause")),
    ("Dead stop, temporarily.", ("unspecified", "none", "unspecified", "pause")),
    ("Quiet for a spell, standing by.", ("unspecified", "none", "unspecified", "pause")),
    ("Hang fire, instructions pending.", ("unspecified", "none", "unspecified", "pause")),

    # --- abort: final. Same register as the pause rows above, finality is the only cue ---
    ("Idle it, we're not continuing.", ("unspecified", "none", "unspecified", "abort")),
    ("Sit on your hands, that's the lot.", ("unspecified", "none", "unspecified", "abort")),
    ("Nothing further -- the run's dead.", ("unspecified", "none", "unspecified", "abort")),
    ("Ease off for good, keep out of view.", ("unspecified", "none", "unspecified", "abort")),
    ("Put a pin in it permanently.", ("unspecified", "none", "unspecified", "abort")),
    ("Dead stop, that's us finished.", ("unspecified", "none", "unspecified", "abort")),
    ("Quiet from here on, mission's off.", ("unspecified", "none", "unspecified", "abort")),
    ("Hang fire indefinitely, we've lost the window.", ("unspecified", "none", "unspecified", "abort")),

    # --- resume: bare idioms, the shape that produced hallucinated retry_* actions ---
    ("Roll again.", ("unspecified", "none", "unspecified", "resume")),
    ("Spin it up again.", ("unspecified", "none", "unspecified", "resume")),
    ("Let's have you moving.", ("unspecified", "none", "unspecified", "resume")),
    ("Clear to proceed, same job.", ("unspecified", "none", "unspecified", "resume")),
    ("Wind it back up.", ("unspecified", "none", "unspecified", "resume")),
    ("Nothing holding you now.", ("unspecified", "none", "unspecified", "resume")),

    # --- retry_read vs read_chip: repetition marker is the only difference ---
    ("Scan the tag once more.", ("unspecified", "none", "unspecified", "retry_read")),
    ("That scan didn't take -- go again.", ("unspecified", "none", "unspecified", "retry_read")),
    ("Second attempt on the chip, please.", ("unspecified", "none", "unspecified", "retry_read")),
    ("Read the tag on the blue block.", ("blue", "none", "unspecified", "read_chip")),
    ("Chip number off the red die, nothing else.", ("red", "none", "unspecified", "read_chip")),
    ("I want the tag on the black cube in the northeast.", ("black", "none", "NE", "read_chip")),

    # --- retry_send vs read_chip: "send" appears in both, v11 confused exactly this ---
    ("Push that transmission out again.", ("unspecified", "none", "unspecified", "retry_send")),
    ("The upload dropped -- one more go.", ("unspecified", "none", "unspecified", "retry_send")),
    ("Transmit the code a second time.", ("unspecified", "none", "unspecified", "retry_send")),
    ("Send me the tag off the yellow target.", ("yellow", "none", "unspecified", "read_chip")),

    # --- report_status vs read_chip: asking about the rover, not a target ---
    ("Where are you at?", ("unspecified", "none", "unspecified", "report_status")),
    ("Give me a sitrep.", ("unspecified", "none", "unspecified", "report_status")),
    ("What's your state?", ("unspecified", "none", "unspecified", "report_status")),
    ("Talk to me -- how's the run tracking?", ("unspecified", "none", "unspecified", "report_status")),

    # --- return_to_start vs abort: motion verb decides, per round 9 ---
    ("Make your way in and shut off.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Drive home, then idle.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Come on back to the box.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Freeze on the spot, nothing more.", ("unspecified", "none", "unspecified", "abort")),
    ("Don't move and don't come in, we're through.", ("unspecified", "none", "unspecified", "abort")),

    # --- collect_target: bare locative is a tasking, plus the avoid_regions constraint ---
    ("Blue cube, northeast quarter.", ("blue", "none", "NE", "collect_target")),
    ("The red block has turned up southwest.", ("red", "none", "SW", "collect_target")),
    ("Fetch the yellow die out of the southeast.", ("yellow", "none", "SE", "collect_target")),
    ("Grab the black object from the northwest.", ("black", "none", "NW", "collect_target")),
    # v3 RELABEL, and the schema's one known expressiveness gap. Every one of these four
    # avoids an UNNAMED region -- "the marked zones", "the shaded areas", "no-go areas",
    # "restricted patches" -- and avoid_region has no value for that: the decision was to
    # drop a separate "unnamed" member and let "none" cover it. So the keep-out constraint
    # is genuinely DISCARDED on these four rows, which is lossy and is written down here
    # rather than hidden.
    #
    # Two of them were also mislabelled under v2 in a way v3 exposes: "from the northeast"
    # and "in the southwest" are the TARGET's quadrant, but v2 put the region in
    # target_location on every avoid_regions row, so those quadrants were being scored as
    # avoided areas. They are targets. That also corrects what I reported earlier about
    # these four: the model was not only facing novel wording, it was facing labels that
    # conflated two different claims.
    #
    # If judges really do say "avoid the shaded areas", avoid_region needs an "unnamed"
    # value and these four rows are the evidence for adding it.
    ("Bring in the blue target, steer clear of the marked zones.",
     ("blue", "none", "unspecified", "collect_target")),
    ("Red die from the northeast -- keep off the shaded areas.",
     ("red", "none", "NE", "collect_target")),
    ("Collect the yellow block, no-go areas are off limits.",
     ("yellow", "none", "unspecified", "collect_target")),
    ("Black cube in the southwest, avoid the restricted patches.",
     ("black", "none", "SW", "collect_target")),

    # --- position correction folded into collect_target (round 8) ---
    ("Amend that -- blue die is actually southeast.", ("blue", "none", "SE", "collect_target")),
    ("Scratch the last position, red cube is northwest.", ("red", "none", "NW", "collect_target")),
    ("New fix on the yellow object: northeast.", ("yellow", "none", "NE", "collect_target")),
]


def main():
    build([(t, l, "familiar") for t, l in FAMILIAR]
          + [(t, l, "contested") for t, l in CONTESTED]
          + [(t, l, "novel") for t, l in NOVEL],
          "data/heldout_set.jsonl", {})

    # heldout2 is also checked against heldout1 -- two "clean" sets that share sentences
    # would just be one set scored twice
    build([(t, l, "heldout2") for t, l in HELDOUT2],
          "data/heldout2_set.jsonl", {"heldout1": "data/heldout_set.jsonl"})

    build([(t, l, "heldout3") for t, l in HELDOUT3],
          "data/heldout3_set.jsonl",
          {"heldout1": "data/heldout_set.jsonl", "heldout2": "data/heldout2_set.jsonl"})

    build([(t, l, "heldout4") for t, l in HELDOUT4],
          "data/heldout4_set.jsonl",
          {"heldout1": "data/heldout_set.jsonl", "heldout2": "data/heldout2_set.jsonl",
           "heldout3": "data/heldout3_set.jsonl"})

    build([(t, l, "heldout5") for t, l in HELDOUT5],
          "data/heldout5_set.jsonl",
          {"heldout1": "data/heldout_set.jsonl", "heldout2": "data/heldout2_set.jsonl",
           "heldout3": "data/heldout3_set.jsonl", "heldout4": "data/heldout4_set.jsonl"})

    # heldout6 already carries its own stratum tag per row, so it is passed straight through.
    build(HELDOUT6, "data/heldout6_set.jsonl",
          {f"heldout{i}": f"data/heldout{i}_set.jsonl".replace("heldout1_", "heldout_")
           for i in range(1, 6)})

    # heldout7 replaces heldout6 as the clean instrument: round 11's templates are written
    # from v13's heldout6 error dump, which spends that set. Checked against all six.
    build(HELDOUT7, "data/heldout7_set.jsonl",
          {f"heldout{i}": f"data/heldout{i}_set.jsonl".replace("heldout1_", "heldout_")
           for i in range(1, 7)})

    # heldout8 replaces heldout7: round 12's templates are informed by v14's heldout7 error
    # dump, which spends it. Checked against all seven.
    build(HELDOUT8, "data/heldout8_set.jsonl",
          {f"heldout{i}": f"data/heldout{i}_set.jsonl".replace("heldout1_", "heldout_")
           for i in range(1, 8)})


if __name__ == "__main__":
    main()
