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

# label tuple = (target_color, constraints, target_location, action)

FAMILIAR = [
    # --- avoid_objects, canonical prohibition wording ---
    ("Bring the blue die in and don't make contact with anything else out there.", ("blue", "avoid_objects", "unspecified", "retrieve")),
    ("The red cube is the pickup -- everything else stays untouched.", ("red", "avoid_objects", "unspecified", "retrieve")),
    ("Haul the yellow block back without brushing the other pieces.", ("yellow", "avoid_objects", "unspecified", "retrieve")),
    ("Black target only. Do not touch the rest.", ("black", "avoid_objects", "unspecified", "retrieve")),
    ("Collect the blue object and keep the other dice exactly as they are.", ("blue", "avoid_objects", "unspecified", "retrieve")),
    ("Take the red die home, nothing else on that field gets moved.", ("red", "avoid_objects", "unspecified", "retrieve")),
    ("Yellow is the pickup; the remaining objects must not be touched.", ("yellow", "avoid_objects", "unspecified", "retrieve")),
    ("Get the black cube without laying a finger on the others.", ("black", "avoid_objects", "unspecified", "retrieve")),
    ("Retrieve the blue block, and no contact with the rest of the field.", ("blue", "avoid_objects", "unspecified", "retrieve")),
    ("Red object comes in clean -- do not disturb the others.", ("red", "avoid_objects", "unspecified", "retrieve")),
    ("Pull the yellow die out without shifting any of the other objects.", ("yellow", "avoid_objects", "unspecified", "retrieve")),
    ("Black cube is the objective, everything else is hands off.", ("black", "avoid_objects", "unspecified", "retrieve")),
    ("Grab the blue target from the northwest and leave the others alone.", ("blue", "avoid_objects", "NW", "retrieve")),
    ("Red die in the southeast -- take it without touching the rest.", ("red", "avoid_objects", "SE", "retrieve")),
    ("Collect the yellow object in the northeast, no contact with the other pieces.", ("yellow", "avoid_objects", "NE", "retrieve")),
    ("Black block sits in the southwest; bring it in without disturbing anything.", ("black", "avoid_objects", "SW", "retrieve")),

    # --- plain retrieve, no location slot ---
    ("Go get the blue die and bring it home.", ("blue", "none", "unspecified", "retrieve")),
    ("The red object is the target -- collect it.", ("red", "none", "unspecified", "retrieve")),
    ("Bring the yellow cube in.", ("yellow", "none", "unspecified", "retrieve")),
    ("Black block is what we want. Go fetch it.", ("black", "none", "unspecified", "retrieve")),
    ("Your pickup is the blue target.", ("blue", "none", "unspecified", "retrieve")),
    ("Head out and collect the red die.", ("red", "none", "unspecified", "retrieve")),
    ("We want the yellow object retrieved.", ("yellow", "none", "unspecified", "retrieve")),
    ("Go after the black cube and return with it.", ("black", "none", "unspecified", "retrieve")),
    ("Blue is the objective this run.", ("blue", "none", "unspecified", "retrieve")),
    ("Collect the red block and come home.", ("red", "none", "unspecified", "retrieve")),
    ("The yellow die is yours to bring back.", ("yellow", "none", "unspecified", "retrieve")),
    ("Go recover the black object.", ("black", "none", "unspecified", "retrieve")),
    ("Pick up the red cube and head for the start.", ("red", "none", "unspecified", "retrieve")),
    ("The yellow target needs collecting.", ("yellow", "none", "unspecified", "retrieve")),
    ("Blue object -- retrieve and return.", ("blue", "none", "unspecified", "retrieve")),
    ("Bring the black target back to the starting zone.", ("black", "none", "unspecified", "retrieve")),

    # --- retrieve with a location slot ---
    ("The blue die is in the northwest. Go get it.", ("blue", "none", "NW", "retrieve")),
    ("Red cube, southeast corner -- collect it.", ("red", "none", "SE", "retrieve")),
    ("You'll find the yellow object in the northeast.", ("yellow", "none", "NE", "retrieve")),
    ("Black block is sitting in the southwest quadrant.", ("black", "none", "SW", "retrieve")),
    ("Our blue piece is down in the southeast somewhere.", ("blue", "none", "SE", "retrieve")),
    ("The red die is over in the northwest area.", ("red", "none", "NW", "retrieve")),
    ("Pick up the yellow cube in the southwest.", ("yellow", "none", "SW", "retrieve")),
    ("Black object, northeast quarter. Bring it in.", ("black", "none", "NE", "retrieve")),
    ("Head northwest and collect the blue block.", ("blue", "none", "NW", "retrieve")),
    ("The red target waits in the southeast.", ("red", "none", "SE", "retrieve")),
    ("Yellow die is in the northeast corner -- retrieve it.", ("yellow", "none", "NE", "retrieve")),
    ("Go to the southwest and pick up the black cube.", ("black", "none", "SW", "retrieve")),
    ("Blue object is positioned in the northeast.", ("blue", "none", "NE", "retrieve")),
    ("Retrieve the red block from the southwest corner.", ("red", "none", "SW", "retrieve")),
    ("The yellow target is in the northwest quadrant.", ("yellow", "none", "NW", "retrieve")),
    ("Black die, southeast side. Go collect it.", ("black", "none", "SE", "retrieve")),

    # --- object relocated ---
    ("Be advised: the blue die shifted to the southwest.", ("blue", "none", "SW", "retrieve")),
    ("Red target has been repositioned to the northeast.", ("red", "none", "NE", "retrieve")),
    ("The yellow cube is no longer where it was; check the northwest.", ("yellow", "none", "NW", "retrieve")),
    ("Black object moved -- southeast quadrant now.", ("black", "none", "SE", "retrieve")),
    ("New location on the blue target: northeast corner.", ("blue", "none", "NE", "retrieve")),
    ("The red die got bumped over to the southwest.", ("red", "none", "SW", "retrieve")),
    ("Yellow object is now in the southeast section.", ("yellow", "none", "SE", "retrieve")),
    ("Black cube has shifted to the northwest quarter.", ("black", "none", "NW", "retrieve")),
    ("Updated position, blue block: southwest side.", ("blue", "none", "SW", "retrieve")),
    ("The red target now sits in the northwest.", ("red", "none", "NW", "retrieve")),
    ("Yellow die relocated to the northeast area.", ("yellow", "none", "NE", "retrieve")),
    ("Black object's new spot is the southeast corner.", ("black", "none", "SE", "retrieve")),

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
    ("The northwest is closed for this attempt.", ("unspecified", "avoid_regions", "NW", "retrieve")),
    ("Don't route through the southeast quarter.", ("unspecified", "avoid_regions", "SE", "retrieve")),
    ("Northeast corner is off the table now.", ("unspecified", "avoid_regions", "NE", "retrieve")),
    ("Keep the vehicle out of the southwest.", ("unspecified", "avoid_regions", "SW", "retrieve")),
    ("That northwest section is barred.", ("unspecified", "avoid_regions", "NW", "retrieve")),
    ("Nothing goes through the southeast region.", ("unspecified", "avoid_regions", "SE", "retrieve")),
    ("Stay outside the northeast quadrant.", ("unspecified", "avoid_regions", "NE", "retrieve")),
    ("The southwest area is restricted from here.", ("unspecified", "avoid_regions", "SW", "retrieve")),
    ("Hold off the northwest quarter entirely.", ("unspecified", "avoid_regions", "NW", "retrieve")),
    ("Southeast is a no-entry zone this run.", ("unspecified", "avoid_regions", "SE", "retrieve")),
    ("Skip the northeast section on your route.", ("unspecified", "avoid_regions", "NE", "retrieve")),
    ("The southwest corner is blocked off.", ("unspecified", "avoid_regions", "SW", "retrieve")),
    ("Judges closed the northwest region.", ("unspecified", "avoid_regions", "NW", "retrieve")),
    ("Do not pass through the southeast area.", ("unspecified", "avoid_regions", "SE", "retrieve")),
    ("Northeast quarter is out of play.", ("unspecified", "avoid_regions", "NE", "retrieve")),
    ("Keep well outside the southwest section.", ("unspecified", "avoid_regions", "SW", "retrieve")),
]

# The families v5's round-3 templates targeted, in wording neither model has seen.
# If v5's eval gain was memorisation, it shows up here as no advantage.
CONTESTED = [
    # --- avoid_objects framed as PERMISSION granted to the target, not prohibition ---
    ("You may handle the blue die and nothing besides it.", ("blue", "avoid_objects", "unspecified", "retrieve")),
    ("Clearance is for the red cube alone -- bring it in.", ("red", "avoid_objects", "unspecified", "retrieve")),
    ("The yellow object is the single item you're allowed to move.", ("yellow", "avoid_objects", "unspecified", "retrieve")),
    ("Only the black block may be picked up.", ("black", "avoid_objects", "unspecified", "retrieve")),
    ("Permission extends to the blue target and no further.", ("blue", "avoid_objects", "unspecified", "retrieve")),
    ("The red die is the one object cleared for contact.", ("red", "avoid_objects", "unspecified", "retrieve")),
    ("You're authorised for the yellow cube only.", ("yellow", "avoid_objects", "unspecified", "retrieve")),
    ("Black object alone is yours to take.", ("black", "avoid_objects", "unspecified", "retrieve")),
    ("Contact is limited to the blue block.", ("blue", "avoid_objects", "unspecified", "retrieve")),
    ("The red target is the only piece you may lift.", ("red", "avoid_objects", "unspecified", "retrieve")),
    ("Just the yellow die is in scope for handling.", ("yellow", "avoid_objects", "unspecified", "retrieve")),
    ("Restrict contact to the black cube.", ("black", "avoid_objects", "unspecified", "retrieve")),

    # --- directional distractor: a ROUTE direction precedes the target's own location.
    # The route cardinal never shares a compass component with the target corner.
    ("Run the western edge on the way out, the blue die is in the northeast corner.", ("blue", "none", "NE", "retrieve")),
    ("Keep to the north side as you go, then collect the red cube from the southwest.", ("red", "none", "SW", "retrieve")),
    ("Move along the eastern boundary first; the yellow object is in the northwest.", ("yellow", "none", "NW", "retrieve")),
    ("Track the southern edge, then pick up the black block in the northeast.", ("black", "none", "NE", "retrieve")),
    ("Approach up the western flank -- the blue target sits in the southeast.", ("blue", "none", "SE", "retrieve")),
    ("Take the northern route out and grab the red die from the southwest corner.", ("red", "none", "SW", "retrieve")),
    ("Follow the eastern side down, then collect the yellow cube in the northwest.", ("yellow", "none", "NW", "retrieve")),
    ("Cut across the southern half before retrieving the black object in the northeast.", ("black", "none", "NE", "retrieve")),
    ("Stay along the western boundary, the blue block is in the southeast quadrant.", ("blue", "none", "SE", "retrieve")),
    ("Head up the northern edge; the red target is in the southwest.", ("red", "none", "SW", "retrieve")),
    ("Work the eastern margin, then take the yellow die from the northwest corner.", ("yellow", "none", "NW", "retrieve")),
    ("Come in along the southern side and collect the black cube in the northeast.", ("black", "none", "NE", "retrieve")),

    # --- avoid_objects carried by proximity/route language, no "avoid"-family verb ---
    ("Give the other dice plenty of room while you collect the blue block.", ("blue", "avoid_objects", "unspecified", "retrieve")),
    ("Keep a gap from the rest of the objects and bring the red cube in.", ("red", "avoid_objects", "unspecified", "retrieve")),
    ("Swing well wide of the other pieces to reach the yellow die.", ("yellow", "avoid_objects", "unspecified", "retrieve")),
    ("Leave space around the other objects on your way to the black target.", ("black", "avoid_objects", "unspecified", "retrieve")),
    ("Don't crowd the other dice -- collect the blue object and return.", ("blue", "avoid_objects", "unspecified", "retrieve")),
    ("Stay off the other pieces while you pull the red block out.", ("red", "avoid_objects", "unspecified", "retrieve")),
    ("Take a wide line past the other objects and grab the yellow cube.", ("yellow", "avoid_objects", "unspecified", "retrieve")),
    ("Keep separation from the rest of the field and bring the black die home.", ("black", "avoid_objects", "unspecified", "retrieve")),

    # --- the mirror case: GENERIC obstruction language carries NO constraint.
    # Follows eval_set's precedent ("navigate around any obstacles" -> none): only a
    # definite reference to the field's other objects sets avoid_objects.
    ("Collect the blue die, working around whatever is in your way.", ("blue", "none", "unspecified", "retrieve")),
    ("Get the red cube and pick a path through the clutter.", ("red", "none", "unspecified", "retrieve")),
    ("Bring the yellow object in, steering past any obstruction you meet.", ("yellow", "none", "unspecified", "retrieve")),
    ("Retrieve the black block, negotiating any obstacles en route.", ("black", "none", "unspecified", "retrieve")),
    ("Fetch the blue target and handle whatever terrain you hit.", ("blue", "none", "unspecified", "retrieve")),
    ("Go for the red die, routing past anything blocking you.", ("red", "none", "unspecified", "retrieve")),
    ("Collect the yellow cube and deal with obstructions as they come.", ("yellow", "none", "unspecified", "retrieve")),
    ("Bring the black object back, navigating whatever stands in the path.", ("black", "none", "unspecified", "retrieve")),

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
    ("Can you bring the blue die back for me?", ("blue", "none", "unspecified", "retrieve")),
    ("Could you read the tag on the red cube?", ("red", "none", "unspecified", "read_chip")),
    ("Would you head back to the start now?", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Mind grabbing the yellow object from the northeast?", ("yellow", "none", "NE", "retrieve")),
    ("Can we get the black block in without touching the others?", ("black", "avoid_objects", "unspecified", "retrieve")),
    ("Can you keep clear of the southwest corner?", ("unspecified", "avoid_regions", "SW", "retrieve")),

    # --- passive voice ---
    ("The blue target is to be collected and returned.", ("blue", "none", "unspecified", "retrieve")),
    ("The red die's chip is to be scanned.", ("red", "none", "unspecified", "read_chip")),
    ("The northeast quadrant is to be avoided.", ("unspecified", "avoid_regions", "NE", "retrieve")),
    ("All other objects are to be left untouched while the yellow cube is recovered.", ("yellow", "avoid_objects", "unspecified", "retrieve")),
    ("The run is to be aborted immediately.", ("unspecified", "none", "unspecified", "abort")),
    ("The vehicle is to be returned to the start.", ("unspecified", "none", "unspecified", "return_to_start")),

    # --- terse fragments ---
    ("Blue. Northwest. Go.", ("blue", "none", "NW", "retrieve")),
    ("Red cube. Chip only.", ("red", "none", "unspecified", "read_chip")),
    ("Abort.", ("unspecified", "none", "unspecified", "abort")),
    ("Home. Now.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("Yellow, southeast, no contact with the others.", ("yellow", "avoid_objects", "SE", "retrieve")),
    ("Southwest closed.", ("unspecified", "avoid_regions", "SW", "retrieve")),

    # --- subordinate clauses ---
    ("If the blue die is still in the northeast, bring it in.", ("blue", "none", "NE", "retrieve")),
    ("Once you're clear of the start, collect the red object and return.", ("red", "none", "unspecified", "retrieve")),
    ("After the reset, scan the yellow cube's tag.", ("yellow", "none", "unspecified", "read_chip")),
    ("Since the northwest is blocked, plan around it.", ("unspecified", "avoid_regions", "NW", "retrieve")),
    ("Because the others must stay put, take only the black block.", ("black", "avoid_objects", "unspecified", "retrieve")),
    ("Whatever else you see, the black die is the only thing coming back.", ("black", "avoid_objects", "unspecified", "retrieve")),

    # --- indirect / polite ---
    ("I'd like the blue object brought back, please.", ("blue", "none", "unspecified", "retrieve")),
    ("We'd rather you didn't enter the southeast quarter.", ("unspecified", "avoid_regions", "SE", "retrieve")),
    ("Let's have the red die's chip read.", ("red", "none", "unspecified", "read_chip")),
    ("Best you come back to start now.", ("unspecified", "none", "unspecified", "return_to_start")),
    ("I need you to stop everything and stay hidden.", ("unspecified", "none", "unspecified", "abort")),
    ("Please bring the yellow cube in without disturbing the rest.", ("yellow", "avoid_objects", "unspecified", "retrieve")),
]

# Second clean slice, written DURING the v6 training run and before any v6 result existed,
# because heldout_set.jsonl is now spent: round 4's templates were designed against its
# failures, so it measures recall for v6 the way eval_set.jsonl already did for v5. Weighted
# toward the classes round 4 changed (avoid_regions, avoid_objects, abort, read_chip), since
# that is where a rebalance can help or backfire.
HELDOUT2 = [
    # --- avoid_regions, wording distinct from round 4's templates ---
    ("Forget about the northwest for now.", ("unspecified", "avoid_regions", "NW", "retrieve")),
    ("The southeast is a dead zone this run.", ("unspecified", "avoid_regions", "SE", "retrieve")),
    ("Nobody goes into the northeast.", ("unspecified", "avoid_regions", "NE", "retrieve")),
    ("Southwest is a write-off, plan without it.", ("unspecified", "avoid_regions", "SW", "retrieve")),
    ("The northwest has been ruled out.", ("unspecified", "avoid_regions", "NW", "retrieve")),
    ("Pretend the southeast quarter does not exist.", ("unspecified", "avoid_regions", "SE", "retrieve")),
    ("Northeast is off your map now.", ("unspecified", "avoid_regions", "NE", "retrieve")),
    ("The southwest section has been struck.", ("unspecified", "avoid_regions", "SW", "retrieve")),
    ("They pulled the northwest from the field.", ("unspecified", "avoid_regions", "NW", "retrieve")),
    ("Southeast quadrant: hands off the whole area.", ("unspecified", "avoid_regions", "SE", "retrieve")),
    ("The northeast is not yours to drive through.", ("unspecified", "avoid_regions", "NE", "retrieve")),
    ("Southwest is barred to the vehicle.", ("unspecified", "avoid_regions", "SW", "retrieve")),
    ("Your route excludes the northwest entirely.", ("unspecified", "avoid_regions", "NW", "retrieve")),
    ("The southeast strip has been retired.", ("unspecified", "avoid_regions", "SE", "retrieve")),
    ("Northeast is a restricted band this round.", ("unspecified", "avoid_regions", "NE", "retrieve")),
    ("Nothing of yours crosses the southwest.", ("unspecified", "avoid_regions", "SW", "retrieve")),

    # --- avoid_objects, prohibition framing ---
    ("The blue die comes back; nothing near it gets moved.", ("blue", "avoid_objects", "unspecified", "retrieve")),
    ("Red cube only. The rest of the field is untouchable.", ("red", "avoid_objects", "unspecified", "retrieve")),
    ("Fetch the yellow block, and let the other dice be.", ("yellow", "avoid_objects", "unspecified", "retrieve")),
    ("Black target is the sole pickup, the others stay planted.", ("black", "avoid_objects", "unspecified", "retrieve")),
    ("Bring the blue object in without upsetting the arrangement.", ("blue", "avoid_objects", "unspecified", "retrieve")),
    ("Take the red die and let everything else sit.", ("red", "avoid_objects", "unspecified", "retrieve")),
    ("Yellow cube out, nothing else moves an inch.", ("yellow", "avoid_objects", "unspecified", "retrieve")),
    ("Collect the black block; the others are not yours.", ("black", "avoid_objects", "unspecified", "retrieve")),

    # --- avoid_objects, permission framing (still only half-learned by v5) ---
    ("The blue block is the extent of what you may touch.", ("blue", "avoid_objects", "unspecified", "retrieve")),
    ("Your remit covers the red die and nothing else.", ("red", "avoid_objects", "unspecified", "retrieve")),
    ("The yellow object is the whole of your clearance.", ("yellow", "avoid_objects", "unspecified", "retrieve")),
    ("You are cleared for the black cube, full stop.", ("black", "avoid_objects", "unspecified", "retrieve")),
    ("Handling rights: the blue die, and that is all.", ("blue", "avoid_objects", "unspecified", "retrieve")),
    ("The red target is where your permission ends.", ("red", "avoid_objects", "unspecified", "retrieve")),

    # --- avoid_objects, proximity framing, and on a read_chip run ---
    ("Leave a margin around the other dice and collect the yellow block.", ("yellow", "avoid_objects", "unspecified", "retrieve")),
    ("Keep off the rest of the field while you take the black object.", ("black", "avoid_objects", "unspecified", "retrieve")),
    ("Tag the blue cube and let the others alone.", ("blue", "avoid_objects", "unspecified", "read_chip")),
    ("Chip read on the red block, the others stay put.", ("red", "avoid_objects", "unspecified", "read_chip")),

    # --- plain retrieve ---
    ("Blue die, bring it in.", ("blue", "none", "unspecified", "retrieve")),
    ("The red object needs to come home.", ("red", "none", "unspecified", "retrieve")),
    ("Fetch the yellow block.", ("yellow", "none", "unspecified", "retrieve")),
    ("Black cube is the target.", ("black", "none", "unspecified", "retrieve")),
    ("Go and collect the blue target from the northeast.", ("blue", "none", "NE", "retrieve")),
    ("Red die is in the southwest.", ("red", "none", "SW", "retrieve")),
    ("The yellow object sits in the northwest.", ("yellow", "none", "NW", "retrieve")),
    ("Black block, southeast side.", ("black", "none", "SE", "retrieve")),
    ("The blue target has moved to the southwest.", ("blue", "none", "SW", "retrieve")),
    ("Red object is now in the northeast.", ("red", "none", "NE", "retrieve")),
    ("Yellow die shifted to the northwest.", ("yellow", "none", "NW", "retrieve")),
    ("Black cube is now over in the southeast.", ("black", "none", "SE", "retrieve")),

    # --- generic obstruction: the mirror case, still no constraint ---
    ("Collect the blue block, getting past whatever is in the way.", ("blue", "none", "unspecified", "retrieve")),
    ("Bring the red die in, around any junk on the floor.", ("red", "none", "unspecified", "retrieve")),
    ("Fetch the yellow object and cope with the terrain.", ("yellow", "none", "unspecified", "retrieve")),
    ("Get the black cube back, whatever the route throws up.", ("black", "none", "unspecified", "retrieve")),

    # --- directional distractor ---
    ("Run the eastern line out, then take the blue die from the northwest.", ("blue", "none", "NW", "retrieve")),
    ("Hold the southern edge, the red cube is in the northeast.", ("red", "none", "NE", "retrieve")),
    ("Come up the western side and collect the yellow object in the southeast.", ("yellow", "none", "SE", "retrieve")),
    ("Track the northern boundary, then grab the black block in the southwest.", ("black", "none", "SW", "retrieve")),

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

FIELDS = ("target_color", "constraints", "target_location", "action")

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
            row = {"text": text, "expected": dict(zip(FIELDS, labels)), "set": tag}
            f.write(json.dumps(row) + "\n")
    counts = collections.Counter(tag for _, _, tag in tagged)
    mix = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"wrote {len(tagged)} examples to {out_path}  [{mix}]  "
          f"0 exact overlap with {', '.join(sorted(existing))}")


def main():
    build([(t, l, "familiar") for t, l in FAMILIAR]
          + [(t, l, "contested") for t, l in CONTESTED]
          + [(t, l, "novel") for t, l in NOVEL],
          "data/heldout_set.jsonl", {})

    # heldout2 is also checked against heldout1 -- two "clean" sets that share sentences
    # would just be one set scored twice
    build([(t, l, "heldout2") for t, l in HELDOUT2],
          "data/heldout2_set.jsonl", {"heldout1": "data/heldout_set.jsonl"})


if __name__ == "__main__":
    main()
