"""
Explicit, precisely-worded Q&A for the numbered/structured facts in the BIM
domain - the exact spots where the fine-tuned model was observed inventing
plausible-but-wrong specifics (e.g. claiming ISO 19650 "consists of two
parts" or mislabeling LOD 300/500).

Unlike kb_bim.py, these aren't run through the template engine - they're
hand-written and deliberately repeated with light phrasing variation, so a
tiny LoRA fine-tune sees the exact numbers enough times to pin them down
instead of paraphrasing them into something wrong. Each list entry is
(question, answer); duplicated phrasings are intentional oversampling, not
a mistake.
"""

STRUCTURED_FACTS = [
    # ISO 19650 parts - the model previously said "two parts" and invented
    # "Part 4: operation". Correct: published parts are 1, 2, 3, and 5.
    ("What are the parts of ISO 19650?",
     "ISO 19650 is published in four parts, not five: Part 1 (concepts and principles), "
     "Part 2 (delivery phase of assets), Part 3 (operational phase of assets), and Part 5 "
     "(security-minded approach to information management). There is no published Part 4."),
    ("How many parts does ISO 19650 have, and what does each one cover?",
     "ISO 19650 has four published parts: Part 1 covers concepts and principles, Part 2 covers "
     "the delivery phase of assets, Part 3 covers the operational phase of assets, and Part 5 "
     "covers a security-minded approach to information management. There is no Part 4."),
    ("List the parts of ISO 19650 in order.",
     "Part 1: concepts and principles. Part 2: delivery phase of assets. Part 3: operational "
     "phase of assets. Part 5: security-minded approach to information management. Note the "
     "numbering skips Part 4 - it was never published."),
    ("Does ISO 19650 have a Part 4?",
     "No. ISO 19650 is published as Parts 1, 2, 3, and 5 - there is no published Part 4."),

    # LOD scale - the model previously invented labels like "LOD 300 (Basic)"
    # and "LOD 500 (Detail)", which aren't the standard descriptors.
    ("List the Level of Development (LOD) scale from 100 to 500.",
     "LOD 100: conceptual, generic massing. LOD 200: approximate geometry and generic systems. "
     "LOD 300: precise geometry, design development. LOD 350: LOD 300 plus interfaces with other "
     "systems. LOD 400: fabrication-ready detail. LOD 500: as-built record of what was actually "
     "installed."),
    ("What does each LOD level from 100 to 500 mean?",
     "LOD 100 is conceptual/generic massing, LOD 200 is approximate geometry with generic "
     "systems, LOD 300 is precise geometry at design development stage, LOD 350 adds interfaces "
     "with other systems, LOD 400 is fabrication-ready detail, and LOD 500 is the as-built record "
     "of the actual installed elements."),
    ("Is LOD 300 the same as 'fabrication-ready'?",
     "No. LOD 300 means precise geometry at design development stage. Fabrication-ready detail "
     "is LOD 400, not LOD 300."),
    ("What's the difference between LOD 400 and LOD 500?",
     "LOD 400 is fabrication-ready detail, used to drive manufacturing and construction. LOD 500 "
     "is the as-built record of what was actually installed, produced after construction is "
     "complete."),

    # CDE workflow states - four states, fixed order/names.
    ("What are the four CDE workflow states in ISO 19650?",
     "Work in Progress, Shared, Published, and Archive."),
    ("List the CDE states in order and what each means.",
     "Work in Progress: not yet shared with other teams. Shared: checked and available to other "
     "teams for coordination. Published: authorized for a specific use such as construction or "
     "tender. Archive: a fixed record kept at the end of a stage or project."),
]
