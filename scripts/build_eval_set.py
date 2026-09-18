"""Builder for data/eval_set.jsonl.

The original 57 examples (EXAMPLES, tagged "core57") are FROZEN -- never edit or
reorder them, so every number ever reported against them stays comparable.

EXT_EXAMPLES (tagged "ext") was added later to fix coverage: the core set had only
3 read_chip and 2 return_to_start examples, too few to measure either class. These
were written from the rules doc's action categories, deliberately WITHOUT looking at
which core examples the model was failing -- targeting known failures would turn the
held-out set into a training set. Score the two tags separately: "ext" is the honest
generalisation number, "core57" is the continuity number.
"""
import json

# label tuple = (target_color, constraints, target_location, action)
EXAMPLES = [
    # --- canonical: Leader's Intent examples, rules doc section 2.2 ---
    ("Retrieve the blue box and bring it back to the starting zone while avoiding all other objects.",
     ("blue", "avoid_objects", "unspecified", "retrieve")),
    ("Target has changed: return the yellow box to the start safely and ignore everything else.",
     ("yellow", "avoid_objects", "unspecified", "retrieve")),
    ("Your goal is to collect the red box and navigate around any obstacles to deliver it to the starting zone.",
     ("red", "none", "unspecified", "retrieve")),
    # NOTE: source sentence used "green box" as placeholder phrasing; green is not
    # a valid competition color (section 2.7 lists blue/red/yellow/black only),
    # so the color word is swapped to black, phrasing kept intact.
    ("Focus on getting the black box and avoiding contact with every other object on the field.",
     ("black", "avoid_objects", "unspecified", "retrieve")),
    # NOTE: source sentence used "orange box" placeholder; swapped to blue for the same reason.
    ("New tasking: find and bring back the blue box without touching other objects or leaving the safe pathway.",
     ("blue", "avoid_objects", "unspecified", "retrieve")),
    ("Stay to the far south of the field until you reach the eastern half and then obtain the blue object in the northeast corner.",
     ("blue", "none", "NE", "retrieve")),
    ("Abort the mission, remain undetected.",
     ("unspecified", "none", "unspecified", "abort")),
    ("Avoid the southwest quarter of the field.",
     ("unspecified", "avoid_regions", "SW", "retrieve")),
    ("The blue object has been reported to have moved to the northeast quarter of the field.",
     ("blue", "none", "NE", "retrieve")),

    # --- invented: target retrieval ---
    ("Go grab the black die and haul it back to base, steer clear of everything else out there.",
     ("black", "avoid_objects", "unspecified", "retrieve")),
    ("We need the red one. Get it and come straight back without bumping into any of the other objects.",
     ("red", "avoid_objects", "unspecified", "retrieve")),
    ("Can you scan the yellow target's chip for me? You don't need to carry it back this time.",
     ("yellow", "none", "unspecified", "read_chip")),
    ("Just tap the NFC on the blue die and send me the code, leave it where it is.",
     ("blue", "none", "unspecified", "read_chip")),
    ("Head out and secure the black object, then make your way home.",
     ("black", "none", "unspecified", "retrieve")),
    ("Priority target is red -- retrieve it and get back to the starting square, no detours near the other dice.",
     ("red", "avoid_objects", "unspecified", "retrieve")),
    ("Bring me the yellow one. Everything else on the field is off limits.",
     ("yellow", "avoid_objects", "unspecified", "retrieve")),
    ("Objective: blue target, northwest side of the field. Retrieve it and return.",
     ("blue", "none", "NW", "retrieve")),
    ("The black target is sitting in the southeast corner -- go get it and bring it home.",
     ("black", "none", "SE", "retrieve")),
    ("You're cleared to grab the red die in the northeast section, just don't touch anything else.",
     ("red", "avoid_objects", "NE", "retrieve")),
    ("Read the chip off the yellow object in the southwest corner and radio back the code.",
     ("yellow", "none", "SW", "read_chip")),
    ("Mission's done once you're carrying the blue target -- pick it up and return to start.",
     ("blue", "none", "unspecified", "retrieve")),

    # --- invented: abort ---
    ("Scrub the mission. Power down and hold position.",
     ("unspecified", "none", "unspecified", "abort")),
    ("Stand down immediately, we're pulling the plug on this run.",
     ("unspecified", "none", "unspecified", "abort")),
    ("Abort, abort -- do not engage the target.",
     ("unspecified", "none", "unspecified", "abort")),
    ("Cancel the retrieval, get out of there quietly.",
     ("unspecified", "none", "unspecified", "abort")),
    ("We're calling it. Halt all movement and go dark.",
     ("unspecified", "none", "unspecified", "abort")),
    ("Terminate the mission now, don't let anyone see you.",
     ("unspecified", "none", "unspecified", "abort")),
    ("Pull back, the mission is off.",
     ("unspecified", "none", "unspecified", "abort")),
    ("Kill the run, hold your position and stay hidden.",
     ("unspecified", "none", "unspecified", "abort")),
    ("Mission's scrapped -- freeze in place.",
     ("unspecified", "none", "unspecified", "abort")),
    ("Disengage and stay out of sight, we're done here.",
     ("unspecified", "none", "unspecified", "abort")),
    ("Forget the target, just bring the system back to the starting zone.",
     ("unspecified", "none", "unspecified", "return_to_start")),
    ("Call off the retrieval and head back to base immediately.",
     ("unspecified", "none", "unspecified", "return_to_start")),

    # --- invented: avoid-region ---
    ("Stay out of the northeast section of the field.",
     ("unspecified", "avoid_regions", "NE", "retrieve")),
    ("Keep clear of the southeast corner, that area's off limits.",
     ("unspecified", "avoid_regions", "SE", "retrieve")),
    ("Do not enter the northwest quadrant for the rest of the run.",
     ("unspecified", "avoid_regions", "NW", "retrieve")),
    ("The southwest side is a no-go zone, route around it.",
     ("unspecified", "avoid_regions", "SW", "retrieve")),
    ("Steer clear of the northeast corner, judges' equipment is set up there.",
     ("unspecified", "avoid_regions", "NE", "retrieve")),
    ("New restriction: the southeast quarter is now off limits.",
     ("unspecified", "avoid_regions", "SE", "retrieve")),
    ("Avoid the northwest corner entirely from this point forward.",
     ("unspecified", "avoid_regions", "NW", "retrieve")),
    ("That southwest region just got flagged as unsafe -- don't go near it.",
     ("unspecified", "avoid_regions", "SW", "retrieve")),
    ("We need you to keep away from the northeast quadrant for now.",
     ("unspecified", "avoid_regions", "NE", "retrieve")),
    ("There's a hazard in the southeast corner, avoid that whole area.",
     ("unspecified", "avoid_regions", "SE", "retrieve")),
    ("Reroute away from the northwest section, it's restricted.",
     ("unspecified", "avoid_regions", "NW", "retrieve")),
    ("Do not cross into the southwest quarter under any circumstance.",
     ("unspecified", "avoid_regions", "SW", "retrieve")),

    # --- invented: object-relocated ---
    ("Update: the red target has shifted to the southwest corner.",
     ("red", "none", "SW", "retrieve")),
    ("Heads up, the yellow die moved -- it's now in the northwest section.",
     ("yellow", "none", "NW", "retrieve")),
    ("Correction: the black object is now sitting in the southeast quadrant.",
     ("black", "none", "SE", "retrieve")),
    ("The blue target's position changed, it's over in the southeast corner now.",
     ("blue", "none", "SE", "retrieve")),
    ("Target relocated -- red object is now northeast.",
     ("red", "none", "NE", "retrieve")),
    ("The yellow die has been moved to the southwest quarter.",
     ("yellow", "none", "SW", "retrieve")),
    ("New position for the black target: northwest corner of the field.",
     ("black", "none", "NW", "retrieve")),
    ("Just so you know, the blue object shifted over to the northwest side.",
     ("blue", "none", "NW", "retrieve")),
    ("The red target isn't where it was -- it's now in the southeast section.",
     ("red", "none", "SE", "retrieve")),
    ("Object update: yellow target moved to the northeast quadrant.",
     ("yellow", "none", "NE", "retrieve")),
    ("The black die relocated to the southeast corner.",
     ("black", "none", "SE", "retrieve")),
    ("Position change: the blue target is now in the southwest quarter.",
     ("blue", "none", "SW", "retrieve")),
]

# --- coverage extension; see module docstring. Written blind to model failures. ---
EXT_EXAMPLES = [
    # --- read_chip (core set had only 3) ---
    ("Ping the tag on the black cube and send me what it says.",
     ("black", "none", "unspecified", "read_chip")),
    ("I just need the chip ID from the red target in the northwest corner.",
     ("red", "none", "NW", "read_chip")),
    ("Scan the blue die's chip, then leave it exactly where it sits.",
     ("blue", "none", "unspecified", "read_chip")),
    ("Get me a chip read on the yellow object in the northeast quadrant.",
     ("yellow", "none", "NE", "read_chip")),
    ("Don't pick it up -- just read the tag on the black target and report back.",
     ("black", "none", "unspecified", "read_chip")),
    ("Pull the NFC data off the red cube in the southeast section.",
     ("red", "none", "SE", "read_chip")),
    ("Chip read only on the blue one, no retrieval.",
     ("blue", "none", "unspecified", "read_chip")),
    ("Log the tag number from the yellow die and stay clear of the other objects.",
     ("yellow", "avoid_objects", "unspecified", "read_chip")),
    ("Read out the chip on the black object sitting in the southwest area.",
     ("black", "none", "SW", "read_chip")),
    ("We only need telemetry from the red target's tag this run.",
     ("red", "none", "unspecified", "read_chip")),
    ("Scan, don't collect -- the blue block in the northwest corner.",
     ("blue", "none", "NW", "read_chip")),
    ("Capture the chip code from the yellow target and hold position.",
     ("yellow", "none", "unspecified", "read_chip")),
    ("Take a tag reading on the black die in the northeast corner without touching anything else.",
     ("black", "avoid_objects", "NE", "read_chip")),
    ("Just the chip data from the red object in the southwest quarter.",
     ("red", "none", "SW", "read_chip")),

    # --- return_to_start (core set had only 2) ---
    ("Abandon the target and drive back to the start line.",
     ("unspecified", "none", "unspecified", "return_to_start")),
    ("Come home, we're not finishing this one.",
     ("unspecified", "none", "unspecified", "return_to_start")),
    ("Send the vehicle back to the starting zone now.",
     ("unspecified", "none", "unspecified", "return_to_start")),
    ("Leave it -- return to the start box.",
     ("unspecified", "none", "unspecified", "return_to_start")),
    ("Navigate the system back to where it launched from.",
     ("unspecified", "none", "unspecified", "return_to_start")),
    ("Wrap it up and bring the rover back to start.",
     ("unspecified", "none", "unspecified", "return_to_start")),
    ("Return to base, empty-handed is fine.",
     ("unspecified", "none", "unspecified", "return_to_start")),
    ("Back to the starting square, we're resetting.",
     ("unspecified", "none", "unspecified", "return_to_start")),
    ("Disregard the objective and drive home.",
     ("unspecified", "none", "unspecified", "return_to_start")),
    ("Bring it back to the start, we'll try again after the reset.",
     ("unspecified", "none", "unspecified", "return_to_start")),
    ("Route the vehicle back to the starting zone and park it.",
     ("unspecified", "none", "unspecified", "return_to_start")),
    ("Head home now, leave the target where it is.",
     ("unspecified", "none", "unspecified", "return_to_start")),
    ("Recall to start -- nothing else needed this run.",
     ("unspecified", "none", "unspecified", "return_to_start")),
    ("Give up on the object and return to the launch point.",
     ("unspecified", "none", "unspecified", "return_to_start")),

    # --- abort ---
    ("Hard stop. Do not move and do not transmit.",
     ("unspecified", "none", "unspecified", "abort")),
    ("Full stop -- remain where you are and keep out of sight.",
     ("unspecified", "none", "unspecified", "abort")),
    ("Everything off, we've lost clearance to run.",
     ("unspecified", "none", "unspecified", "abort")),
    ("Hold everything. No movement until I say otherwise.",
     ("unspecified", "none", "unspecified", "abort")),
    ("The attempt is void -- stop and conceal.",
     ("unspecified", "none", "unspecified", "abort")),

    # --- retrieve, including compound "do X and come back" phrasing ---
    ("Collect the blue cube from the southwest corner and come back around the other objects.",
     ("blue", "avoid_objects", "SW", "retrieve")),
    ("Grab the yellow block in the northeast and bring it home.",
     ("yellow", "none", "NE", "retrieve")),
    ("Your objective is the black die in the northwest quadrant.",
     ("black", "none", "NW", "retrieve")),
    ("Secure the red object and return -- nothing else on the field may be touched.",
     ("red", "avoid_objects", "unspecified", "retrieve")),
    ("Retrieve the blue target from the southeast section.",
     ("blue", "none", "SE", "retrieve")),
    ("The yellow one needs to come back with you.",
     ("yellow", "none", "unspecified", "retrieve")),
    ("Take the black cube back to the starting zone, keep away from the other pieces.",
     ("black", "avoid_objects", "unspecified", "retrieve")),
    ("Go get the red die in the northwest area and haul it in.",
     ("red", "none", "NW", "retrieve")),
    ("Blue target, southwest corner -- bring it to start.",
     ("blue", "none", "SW", "retrieve")),
    ("Recover the yellow object without contacting any other item on the field.",
     ("yellow", "avoid_objects", "unspecified", "retrieve")),
    ("Head to the southeast and collect the black target.",
     ("black", "none", "SE", "retrieve")),
    ("Pick up the red cube in the northeast and carry it to the start line.",
     ("red", "none", "NE", "retrieve")),

    # --- avoid_regions ---
    ("The northeast corner is closed off for this run.",
     ("unspecified", "avoid_regions", "NE", "retrieve")),
    ("Give the southwest quadrant a wide berth.",
     ("unspecified", "avoid_regions", "SW", "retrieve")),
    ("Northwest is out of bounds now.",
     ("unspecified", "avoid_regions", "NW", "retrieve")),
    ("Nothing may enter the southeast region.",
     ("unspecified", "avoid_regions", "SE", "retrieve")),
    ("Treat the northeast quarter as restricted -- stay outside it.",
     ("unspecified", "avoid_regions", "NE", "retrieve")),
    ("The southwest section stays off the route from here on.",
     ("unspecified", "avoid_regions", "SW", "retrieve")),
]

FIELDS = ("target_color", "constraints", "target_location", "action")

def main():
    tagged = [(t, l, "core57") for t, l in EXAMPLES] + [(t, l, "ext") for t, l in EXT_EXAMPLES]

    seen = {}
    for text, _, tag in tagged:
        if text in seen:
            raise SystemExit(f"duplicate eval text ({seen[text]} vs {tag}): {text!r}")
        seen[text] = tag

    with open("data/eval_set.jsonl", "w", encoding="utf-8") as f:
        for text, labels, tag in tagged:
            row = {"text": text, "expected": dict(zip(FIELDS, labels)), "set": tag}
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(tagged)} examples to data/eval_set.jsonl "
          f"({len(EXAMPLES)} core57 + {len(EXT_EXAMPLES)} ext)")

if __name__ == "__main__":
    main()
