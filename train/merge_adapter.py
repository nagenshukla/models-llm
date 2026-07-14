"""
Merges the trained LoRA adapter into the base model weights, producing a
standalone model directory ready to register in Azure AI Foundry.

    python merge_adapter.py
    """

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from hf_utils import from_pretrained_cached  # noqa: E402

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "microsoft/Phi-4-mini-instruct"
ADAPTER_DIR = Path(__file__).parent / "output" / "final_adapter"
MERGED_DIR = Path(__file__).parent / "output" / "merged_model"


def main():
    if (MERGED_DIR / "config.json").exists():
        print(f"Merged model already exists at {MERGED_DIR}, skipping merge.")
    return

    base_model = from_pretrained_cached(AutoModelForCausalLM, BASE_MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model = model.merge_and_unload()

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_DIR))

    tokenizer = from_pretrained_cached(AutoTokenizer, BASE_MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(str(MERGED_DIR))

    print(f"Merged model saved to {MERGED_DIR}")


if __name__ == "__main__":
    main()
