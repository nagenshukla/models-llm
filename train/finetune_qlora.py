"""
QLoRA supervised fine-tune of a small instruct model on the synthetic BIM
dataset (../data/train.jsonl).

Run under WSL2 or a Linux/CUDA Docker container - bitsandbytes 4-bit
quantization and flash-attention are unreliable on native Windows.

    pip install -r requirements.txt
    python finetune_qlora.py

Produces a LoRA adapter in ./output/. Merge it into the base model with
merge_adapter.py before packaging for deployment.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hf_utils import from_pretrained_cached  # noqa: E402

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import torch

BASE_MODEL = "microsoft/Phi-4-mini-instruct"
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"


def load_jsonl_as_dataset(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)


def format_chat(example, tokenizer):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": text}


def main():
    train_ds = load_jsonl_as_dataset(DATA_DIR / "train.jsonl")
    eval_ds = load_jsonl_as_dataset(DATA_DIR / "test.jsonl")

    # Phi-4-mini isn't in every transformers release's native model mapping,
    # so it needs the repo's bundled remote code regardless of this flag -
    # hf_utils patches the transformers-version skew that code hits (see
    # _patch_loss_kwargs_compat in hf_utils.py) rather than fighting it here.
    tokenizer = from_pretrained_cached(AutoTokenizer, BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = train_ds.map(lambda ex: format_chat(ex, tokenizer))
    eval_ds = eval_ds.map(lambda ex: format_chat(ex, tokenizer))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = from_pretrained_cached(
        AutoModelForCausalLM,
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final_adapter"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final_adapter"))
    print(f"Adapter saved to {OUTPUT_DIR / 'final_adapter'}")


if __name__ == "__main__":
    main()
