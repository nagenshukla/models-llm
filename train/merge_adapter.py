"""
Merges the trained LoRA adapter into the base model weights, producing a
standalone model directory ready to register in Azure AI Foundry.

    python merge_adapter.py
"""

from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "microsoft/Phi-4-mini-instruct"
ADAPTER_DIR = Path(__file__).parent / "output" / "final_adapter"
MERGED_DIR = Path(__file__).parent / "output" / "merged_model"


def main():
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=False)
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model = model.merge_and_unload()

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_DIR))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False)
    tokenizer.save_pretrained(str(MERGED_DIR))

    print(f"Merged model saved to {MERGED_DIR}")


if __name__ == "__main__":
    main()
