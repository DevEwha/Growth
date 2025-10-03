#!/usr/bin/env python3
"""
LoRA-only QA finetuning script (no QLoRA), designed for pruned local models.
- Works directly with a pre-pruned HF folder (e.g., ./hybrid_sparse35).
- Masked-label generative QA training (prompt tokens -> -100, learn to generate answer only).
- Full-precision (bf16/fp16) by default for max accuracy; optional 8-bit base for memory saving.

Usage (examples):

# 1) LoRA on pruned base (full precision), SQuAD train for 1 epoch
python lora_qa_train.py \
  --model_path ./hybrid_sparse35 \
  --tokenizer_path meta-llama/Llama-2-7b-hf \
  --dataset squad --split train \
  --output_dir ./hybrid_sparse35-lora-squad \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 512 --answer_max_length 64 \
  --num_train_epochs 1 \
  --learning_rate 2e-4 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05

# 2) Same but with 8-bit base weights to reduce VRAM (slightly slower than fp16 but still accurate)
python lora_qa_train.py \
  --model_path ./hybrid_sparse35 \
  --tokenizer_path meta-llama/Llama-2-7b-hf \
  --dataset squad --split train \
  --output_dir ./hybrid_sparse35-lora-squad-8bit \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 512 --answer_max_length 64 \
  --num_train_epochs 1 \
  --learning_rate 2e-4 \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --load_in_8bit

# 3) Quick sanity run (limit tokens processed)
python lora_qa_train.py \
  --model_path ./hybrid_sparse35 \
  --dataset squad --split train \
  --output_dir ./hybrid_sparse35-lora-squad-dry \
  --max_steps 500 --save_steps 250 --logging_steps 25

After training, evaluate with your qa_eval.py:
python qa_eval.py \
  --model_path ./hybrid_sparse35 \
  --adapter_path ./hybrid_sparse35-lora-squad \
  --tokenizer_path meta-llama/Llama-2-7b-hf \
  --dataset squad --split validation --merge_and_unload --fast_preset
"""

import os
import argparse
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import LoraConfig, get_peft_model

# Optional: 8-bit base
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False


# -----------------------
# Prompt & tokenization
# -----------------------
TEMPLATE = """
### 질문:
{question}

### 문맥:
{context}

### 답변:
""".strip()

def make_prompt(q: str, c: str) -> str:
    return TEMPLATE.format(question=q, context=c)


def build_qa_labels(
    examples: Dict[str, List[Any]],
    tokenizer,
    max_length: int,
    answer_max_length: int,
):
    # Robust field extraction (SQuAD-like)
    questions = examples.get("question", examples.get("Question", []))
    contexts  = examples.get("context", [""] * len(questions))
    answers   = examples.get("answers", None)

    # normalize answers to list[str]
    refs: List[List[str]] = []
    if answers is not None:
        for a in answers:
            if isinstance(a, dict):
                txts = a.get("text", [])
            elif isinstance(a, list):
                txts = a
            else:
                txts = []
            if not txts:
                txts = [""]
            refs.append([str(txts[0])])
    else:
        # try other fields
        refs = [[""] for _ in range(len(questions))]

    input_ids_list, attention_mask_list, labels_list = [], [], []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    for q, c, ref_list in zip(questions, contexts, refs):
        target = ref_list[0] if ref_list else ""
        prompt = make_prompt(str(q), str(c))
        target_text = str(target).strip()

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        if len(target_ids) > answer_max_length:
            target_ids = target_ids[:answer_max_length]

        # ensure target kept; trim prompt from the left if needed
        available_for_prompt = max_length - len(target_ids)
        if available_for_prompt <= 0:
            prompt_ids = []
            target_ids = target_ids[:max_length]
            prompt_len = 0
        else:
            if len(prompt_ids) > available_for_prompt:
                prompt_ids = prompt_ids[-available_for_prompt:]
            prompt_len = len(prompt_ids)

        full_ids = prompt_ids + target_ids
        attn = [1] * len(full_ids)
        # pad to max_length (static padding helps throughput; change to dynamic if preferred)
        pad_len = max_length - len(full_ids)
        if pad_len > 0:
            full_ids += [pad_id] * pad_len
            attn     += [0] * pad_len

        labels = [-100] * prompt_len + target_ids + [-100] * (max_length - prompt_len - len(target_ids))
        if len(labels) != max_length:
            labels = labels[:max_length] + [-100] * max(0, max_length - len(labels))

        input_ids_list.append(full_ids)
        attention_mask_list.append(attn)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


# -----------------------
# Collator (static tensors provided above; default is fine)
# -----------------------

def collate_fn(features: List[Dict[str, Any]]):
    return default_data_collator(features)


# -----------------------
# Main
# -----------------------

def main():
    p = argparse.ArgumentParser()
    # IO
    p.add_argument("--model_path", type=str, required=True, help="Pre-pruned HF model dir (e.g., ./hybrid_sparse35)")
    p.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer source (fallback to model_path if None)")
    p.add_argument("--output_dir", type=str, default="./lora_qa_out")

    # Data
    p.add_argument("--dataset", type=str, default="squad")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--answer_max_length", type=int, default=64)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)

    # Train
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1, help="If >0, override epochs and run for this many steps")

    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--save_total_limit", type=int, default=3)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--ffn_only", action="store_true", help="Apply LoRA to MLP (gate/up/down) only")

    # System
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 if supported")
    p.add_argument("--load_in_8bit", action="store_true", help="Load base in 8-bit (accuracy>QLoRA, VRAM<<fp16)")

    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer
    tok_src = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Model
    torch_dtype = torch.bfloat16 if (args.bf16 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    load_kwargs = dict(device_map="auto", torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    if args.load_in_8bit:
        if not _HAS_BNB:
            raise RuntimeError("bitsandbytes not available; install and retry or remove --load_in_8bit")
        load_kwargs.update(
            quantization_config=BitsAndBytesConfig(load_in_8bit=True)
        )

    print("Loading model (LoRA base, no 4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)

    # Build LoRA config
    if args.ffn_only:
        targets = ["gate_proj", "up_proj", "down_proj"]
    else:
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lconf = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lconf)
    model.print_trainable_parameters()

    # Data
    print(f"Loading dataset: {args.dataset} [{args.split}]")
    raw = load_dataset(args.dataset, split=args.split)
    if args.max_train_samples and args.split == "train":
        raw = raw.select(range(min(len(raw), args.max_train_samples)))
    elif args.max_eval_samples and args.split != "train":
        raw = raw.select(range(min(len(raw), args.max_eval_samples)))

    def map_fn(batch):
        return build_qa_labels(batch, tokenizer, args.max_length, args.answer_max_length)

    ds = raw.map(map_fn, batched=True, remove_columns=raw.column_names)

    # Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else None,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type="cosine",
        fp16=(torch_dtype==torch.float16),
        bf16=(torch_dtype==torch.bfloat16),
        report_to=[],
        remove_unused_columns=False,
    )

    # throughput + stability
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collate_fn,
        tokenizer=tokenizer,
    )

    trainer.train()

    print("Saving LoRA adapters + tokenizer...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
