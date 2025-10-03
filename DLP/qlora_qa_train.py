#!/usr/bin/env python3
import os
import argparse
from typing import Dict, Any, List
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

def extract_answer(ex):
    """Extract answer from QA dataset"""
    if "answers" in ex:
        ans = ex["answers"]
        if isinstance(ans, dict) and "text" in ans and ans["text"]:
            return ans["text"][0]
        elif isinstance(ans, list) and ans:
            return ans[0]
    
    for key in ["answer", "answers_text", "labels"]:
        if key in ex and isinstance(ex[key], str):
            return ex[key]
    return None

def qa_token_stream(tokenizer, dataset_name, split, max_seq_len, token_budget):
    """Generate QA training samples"""
    ds = load_dataset(dataset_name, split=split)
    acc_tokens = 0
    
    for ex in ds:
        if acc_tokens >= token_budget:
            break
            
        question = ex.get("question") or ex.get("query")
        context = ex.get("context") or ex.get("article") or ex.get("passage") or ""
        answer = extract_answer(ex)
        
        if not question or not answer:
            continue

        # QA 프롬프트 생성
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer: "
        prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_seq_len)["input_ids"]
        full_ids = tokenizer(prompt + answer, add_special_tokens=False, truncation=True, max_length=max_seq_len)["input_ids"]
        
        if len(prompt_ids) >= max_seq_len:
            continue
            
        # 프롬프트 부분은 loss 계산에서 제외 (-100 마스킹)
        labels = full_ids.copy()
        for i in range(len(prompt_ids)):
            labels[i] = -100

        acc_tokens += len(full_ids)
        yield {"input_ids": full_ids, "labels": labels}

class QADataset(IterableDataset):
    def __init__(self, tokenizer, dataset_name, split, max_seq_len, token_budget):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.split = split
        self.max_seq_len = max_seq_len
        self.token_budget = token_budget
        
    def __iter__(self):
        yield from qa_token_stream(self.tokenizer, self.dataset_name, self.split, self.max_seq_len, self.token_budget)

class QADataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        max_len = max(len(x) for x in input_ids)

        pad_id = self.tokenizer.pad_token_id
        padded_input_ids, padded_labels, attention_masks = [], [], []
        
        for ids, labs in zip(input_ids, labels):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_id] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }

def main():
    parser = argparse.ArgumentParser(description="QLoRA healing for pruned models")
    
    # 기본 설정
    parser.add_argument("--pruned_model_path", type=str, default="./llama2_7b_hybrid_sgpt_auto", 
                       help="Path to pruned model directory")
    parser.add_argument("--dataset", type=str, default="squad", help="QA dataset name")
    parser.add_argument("--output_dir", type=str, default="./healed_model", help="Output directory")
    
    # 훈련 설정
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--token_budget", type=int, default=2_000_000, help="Training tokens")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    
    # LoRA 설정
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_all", action="store_true", 
                       help="Target all layers (default: FFN only)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🔧 Loading pruned model from: {args.pruned_model_path}")
    
    # 토크나이저 로드
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.pruned_model_path, use_fast=True)
    except:
        print("⚠️ Tokenizer not found in pruned model, using Llama-2-7b tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 4-bit quantization으로 모델 로드
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.pruned_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    print(f"✅ Loaded pruned model with {model.config.num_hidden_layers} layers")

    # k-bit 훈련 준비
    model = prepare_model_for_kbit_training(model)

    # LoRA 설정
    if args.target_all:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = ["gate_proj", "up_proj", "down_proj"]  # FFN만
    
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    print("📊 LoRA configuration:")
    model.print_trainable_parameters()
    
    # 메모리 최적화
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # 데이터셋 준비
    print(f"📚 Preparing {args.dataset} dataset...")
    train_dataset = QADataset(tokenizer, args.dataset, "train", args.max_seq_len, args.token_budget)
    data_collator = QADataCollator(tokenizer)

    # 훈련 스텝 계산
    tokens_per_step = args.max_seq_len * args.batch_size * args.gradient_accumulation_steps
    max_steps = max(args.token_budget // tokens_per_step, 1)
    print(f"📈 Estimated training steps: {max_steps}")

    # 훈련 설정
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=max_steps,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_steps=100,
        gradient_checkpointing=True,
        dataloader_drop_last=False,
        report_to=[],
    )

    # 트레이너 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 훈련 시작
    print("🚀 Starting QLoRA healing...")
    trainer.train()

    # 모델 저장
    print("💾 Saving healed model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ QLoRA healing completed!")

if __name__ == "__main__":
    main()
