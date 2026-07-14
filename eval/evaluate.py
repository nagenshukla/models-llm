"""
Scores a model's answers on data/test.jsonl against the ground-truth
knowledge base terms mentioned in the expected answer, using simple
keyword/coverage matching instead of an LLM judge - since the test set
answers were generated from a KB we own, we can grade objectively.

    python evaluate.py --model ../train/output/merged_model
    python evaluate.py --model microsoft/Phi-4-mini-instruct   # baseline comparison

For each test example, checks whether the model's generated answer
mentions the key terms that appear in the reference answer (case-insensitive,
simple substring match on salient words >= 4 chars, deduped). Reports a
per-example coverage score and an aggregate average - a rough but fast
signal to compare fine-tuned vs base model without needing a hosted judge.
"""

import argparse
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

STOPWORDS = {
    "this", "that", "with", "from", "into", "such", "than", "then", "they",
    "their", "which", "these", "those", "about", "would", "could", "should",
    "using", "used", "over", "each", "your", "have", "been", "were", "when",
}


def salient_terms(text):
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-']+", text)
    return {w.lower() for w in words if len(w) >= 4 and w.lower() not in STOPWORDS}


def coverage_score(reference, candidate):
    ref_terms = salient_terms(reference)
    cand_terms = salient_terms(candidate)
    if not ref_terms:
        return 1.0
    return len(ref_terms & cand_terms) / len(ref_terms)


def load_generate_fn(model_path):
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from hf_utils import from_pretrained_cached

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = from_pretrained_cached(AutoTokenizer, model_path, trust_remote_code=True)
    model = from_pretrained_cached(
        AutoModelForCausalLM, model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )

    def generate(system, user):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()

    return generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path or HF id of model to evaluate")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    generate = load_generate_fn(args.model)

    scores = []
    with open(DATA_DIR / "test.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if args.limit:
        rows = rows[: args.limit]

    for row in rows:
        system = row["messages"][0]["content"]
        user = row["messages"][1]["content"]
        reference = row["messages"][2]["content"]
        candidate = generate(system, user)
        score = coverage_score(reference, candidate)
        scores.append(score)
        print(f"[{score:.2f}] Q: {user}")

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"\nAverage term-coverage score over {len(scores)} examples: {avg:.3f}")


if __name__ == "__main__":
    main()
