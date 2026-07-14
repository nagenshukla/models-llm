"""
Generates a synthetic instruction-tuning dataset for a BIM / digital
construction assistant from the hand-curated knowledge base in kb_bim.py.

No external LLM call is used for generation — everything is templated off
ground-truth facts, so the resulting data is grounded by construction and
free to regenerate. Run:

    python generate_synthetic_data.py

Outputs data/train.jsonl and data/test.jsonl in chat format:
    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
"""

import json
import random
from pathlib import Path

from kb_bim import KB, KB_BY_ID

random.seed(42)

SYSTEM_PROMPT = (
    "You are a BIM and digital construction assistant for civil engineers. "
    "Answer using correct construction and BIM terminology, and be concise and accurate."
)

DEFINITION_TEMPLATES = [
    "What is {term}?",
    "Can you define {term} for me?",
    "In BIM, what does {term} mean?",
    "Explain the term {term}.",
    "I keep seeing '{term}' in project docs — what is it?",
]

ACRONYM_TEMPLATES = [
    "What does {acronym} stand for?",
    "In construction, what's the full form of {acronym}?",
    "What does the acronym {acronym} mean in a BIM context?",
]

PERSONA_TEMPLATES = [
    "As a junior BIM coordinator, I keep hearing about {term}. Can you explain it simply?",
    "I'm a site engineer new to BIM workflows — what is {term} and why does it matter?",
    "I'm onboarding onto a project that uses {term} heavily. Give me a quick briefing.",
    "As a project manager without a technical BIM background, how would you explain {term} to me?",
]

APPLIED_TEMPLATES = [
    "When would I actually use {term} on a project?",
    "Give me a real-world example of {term} in practice.",
    "How is {term} applied on a construction project?",
]

COMPARISON_TEMPLATES = [
    "What's the difference between {term_a} and {term_b}?",
    "How does {term_a} relate to {term_b}?",
    "I keep confusing {term_a} and {term_b} — can you clarify?",
]

CATEGORY_DIFFERENCE_HINTS = {
    ("standard", "standard"): "Both are standards, but they cover different parts of the information management process.",
    ("document", "document"): "Both are project documents, but they serve different roles and are produced by different parties at different stages.",
    ("process", "process"): "Both are processes used during a project, but they happen at different points and serve different purposes.",
    ("role", "role"): "Both are project roles, but they operate at different levels of responsibility.",
    ("software", "software"): "Both are software tools used in BIM workflows, but they serve different functions.",
    ("dimension", "dimension"): "Both are BIM 'dimensions', but they add different kinds of information to the model.",
}


def difference_hint(entry_a, entry_b):
    key = (entry_a["category"], entry_b["category"])
    if key in CATEGORY_DIFFERENCE_HINTS:
        return CATEGORY_DIFFERENCE_HINTS[key]
    return "They come from different parts of the BIM process, so they aren't interchangeable."


def make_example(user_text, assistant_text):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def gen_definition_examples(entry):
    examples = []
    term_display = entry["term"]
    if entry["acronym"]:
        term_display += f" ({entry['acronym']})"
    answer = f"{term_display}: {entry['definition']}"
    for tmpl in DEFINITION_TEMPLATES:
        q = tmpl.format(term=entry["term"])
        examples.append(make_example(q, answer))
    return examples


def gen_acronym_examples(entry):
    if not entry["acronym"]:
        return []
    examples = []
    answer = (f"{entry['acronym']} stands for {entry['term']}. "
              f"{entry['definition']}")
    for tmpl in ACRONYM_TEMPLATES:
        q = tmpl.format(acronym=entry["acronym"])
        examples.append(make_example(q, answer))
    return examples


def gen_persona_examples(entry):
    examples = []
    answer = f"{entry['definition']} For example: {entry['example']}"
    for tmpl in PERSONA_TEMPLATES:
        q = tmpl.format(term=entry["term"])
        examples.append(make_example(q, answer))
    return examples


def gen_applied_examples(entry):
    examples = []
    for tmpl in APPLIED_TEMPLATES:
        q = tmpl.format(term=entry["term"])
        examples.append(make_example(q, entry["example"]))
    return examples


def gen_comparison_examples(entry):
    examples = []
    for related_id in entry["related"]:
        other = KB_BY_ID.get(related_id)
        if not other:
            continue
        answer = (
            f"{entry['term']}: {entry['definition']} "
            f"{other['term']}, on the other hand: {other['definition']} "
            f"{difference_hint(entry, other)}"
        )
        tmpl = random.choice(COMPARISON_TEMPLATES)
        q = tmpl.format(term_a=entry["term"], term_b=other["term"])
        examples.append(make_example(q, answer))
    return examples


def generate_all():
    all_examples = []
    for entry in KB:
        all_examples.extend(gen_definition_examples(entry))
        all_examples.extend(gen_acronym_examples(entry))
        all_examples.extend(gen_persona_examples(entry))
        all_examples.extend(gen_applied_examples(entry))
        all_examples.extend(gen_comparison_examples(entry))
    return all_examples


def dedupe(examples):
    seen = set()
    unique = []
    for ex in examples:
        key = ex["messages"][1]["content"] + "||" + ex["messages"][2]["content"]
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def split_train_test(examples, test_fraction=0.15):
    random.shuffle(examples)
    n_test = max(1, int(len(examples) * test_fraction))
    test = examples[:n_test]
    train = examples[n_test:]
    return train, test


def write_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    examples = generate_all()
    examples = dedupe(examples)
    train, test = split_train_test(examples)

    out_dir = Path(__file__).parent
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")

    print(f"KB terms: {len(KB)}")
    print(f"Total examples after dedupe: {len(examples)}")
    print(f"Train: {len(train)} -> {out_dir / 'train.jsonl'}")
    print(f"Test:  {len(test)} -> {out_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
