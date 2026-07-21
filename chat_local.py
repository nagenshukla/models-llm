"""
Quick local sanity-check chat against a local model directory (e.g. the
merged fine-tune downloaded from Colab) - no Foundry, no Colab, no vLLM
required. Runs on CPU by default, so it's slow (expect well over a minute
per response on a 3.8B model), but it's the most compatibility-risk-free
way to confirm the fine-tune actually works before spending money/effort
on hosting it anywhere.

    python chat_local.py "C:/Development/model-weights/merged_model" "What is BIM?"

Omit the question to drop into an interactive loop.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hf_utils import from_pretrained_cached  # noqa: E402

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
import torch  # noqa: E402

SYSTEM_PROMPT = (
    "You are a BIM and digital construction assistant for civil engineers. "
    "Answer using correct construction and BIM terminology, and be concise and accurate."
)


def load(model_path):
    print(f"Loading {model_path} on CPU (this can take a few minutes)...")
    # trust_remote_code omitted here: config.json's auto_map points AutoTokenizer
    # at a bare repo id ("Xenova/gpt-4o"), which crashes transformers' dynamic-module
    # loader. Not needed anyway - tokenizer_class is the built-in GPT2Tokenizer.
    tokenizer = from_pretrained_cached(AutoTokenizer, model_path)
    model = from_pretrained_cached(
        AutoModelForCausalLM, model_path, trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    return tokenizer, model


def ask(tokenizer, model, question):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    if len(sys.argv) < 2:
        print('Usage: python chat_local.py "<model_path>" ["question"]')
        sys.exit(1)

    model_path = sys.argv[1]
    tokenizer, model = load(model_path)

    if len(sys.argv) > 2:
        question = " ".join(sys.argv[2:])
        print(ask(tokenizer, model, question))
        return

    print("Interactive mode - type a question, or 'quit' to exit.")
    while True:
        question = input("\n> ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        print(ask(tokenizer, model, question))


if __name__ == "__main__":
    main()
