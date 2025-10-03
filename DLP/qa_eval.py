#!/usr/bin/env python3
"""
qa_eval.py
Generative QA evaluation for causal LMs (SQuAD-style).
Produces EM / F1 and writes predictions to JSONL.

Usage examples:

# 1) 프루닝된 베이스 + LoRA 어댑터 (권장)
python qa_eval.py \
  --model_path ./hybrid_sparse35 \
  --adapter_path ./hybrid_sparse35-qlora-squad/pruned_healed_peft \
  --tokenizer_path meta-llama/Llama-2-7b-hf \
  --dataset squad --split validation \
  --max_eval_samples 200 --batch_size 4 --merge_and_unload

# 2) 어댑터 폴더만 있을 때 (peft_config.json 포함)
python qa_eval.py \
  --model_path ./hybrid_sparse35-qlora-squad/pruned_healed_peft \
  --tokenizer_path meta-llama/Llama-2-7b-hf \
  --dataset squad --split validation --max_eval_samples 200

# 3) 머지된 단일 모델 폴더만 있을 때
python qa_eval.py \
  --model_path ./hybrid_sparse35-qlora-squad-merged \
  --dataset squad --split validation --max_eval_samples 200
"""

import argparse
import json
import re
import string
from typing import List, Dict, Any, Optional
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc

# ---- Dataset configurations ----
DATASET_CONFIGS = {
    "squad": {"question_key": "question", "context_key": "context", "answer_key": "answers"},
    "korquad": {"question_key": "question", "context_key": "context", "answer_key": "answers"},
    "squad_ko_v1": {"question_key": "question", "context_key": "context", "answer_key": "answers"},
    "custom_qa": {"question_key": "query", "context_key": "passage", "answer_key": "answer"},
    "ms_marco": {"question_key": "query", "context_key": "passages", "answer_key": "answers"}
}

# ---- metrics (normalize / EM / F1) ----
def normalize_answer(s: str) -> str:
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):  return ' '.join(text.split())
    def remove_punc(text):      return ''.join(ch for ch in text if ch not in set(string.punctuation))
    s = (s or "").lower()
    s = remove_punc(s); s = remove_articles(s); s = white_space_fix(s)
    return s.strip()

def f1_score(pred: str, truth: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    truth_tokens = normalize_answer(truth).split()
    if not pred_tokens and not truth_tokens: return 1.0
    if not pred_tokens or not truth_tokens:  return 0.0
    common = {}
    for t in pred_tokens: common[t] = common.get(t, 0) + 1
    matches = 0
    for t in truth_tokens:
        if common.get(t, 0) > 0:
            matches += 1; common[t] -= 1
    if matches == 0: return 0.0
    precision = matches / len(pred_tokens)
    recall    = matches / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def exact_match_score(pred: str, truth: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(truth) else 0.0

def metric_best(pred: str, truths: List[str]) -> Dict[str, float]:
    truths = truths or [""]
    best_em = best_f1 = 0.0
    for t in truths:
        best_em  = max(best_em,  exact_match_score(pred, t))
        best_f1  = max(best_f1,  f1_score(pred, t))
    return {"em": best_em, "f1": best_f1}

# ---- prompt templates ----
TEMPLATES = {
    "simple": "Context: {context}\n\nQuestion: {question}\n\nAnswer:",
    "basic": "{context}\n\nQ: {question}\nA:",
    "squad_style": "context: {context}\nquestion: {question}\nanswer:",
}



def build_prompt(template_name: str, question: str, context: str) -> str:
    return TEMPLATES.get(template_name, TEMPLATES["squad_style"]).format(question=question, context=context)

def load_tokenizer(tokenizer_path: Optional[str], model_path: Optional[str], 
                  adapter_path: Optional[str], max_length: int):
    """토크나이저 로드 (폴백 순서: tokenizer_path -> model_path -> adapter_path)"""
    # 우선순위: 명시적으로 지정된 토크나이저 경로 먼저
    tried_paths = []
    if tokenizer_path:
        tried_paths.append(tokenizer_path)
    if model_path:
        tried_paths.append(model_path)
    if adapter_path:
        tried_paths.append(adapter_path)
    
    last_err = None
    tok = None
    
    for p in tried_paths:
        try:
            print(f"Trying to load tokenizer from: {p}")
            tok = AutoTokenizer.from_pretrained(p, use_fast=True, trust_remote_code=True)
            print(f"✓ Tokenizer loaded from: {p}")
            break
        except Exception as e:
            print(f"❌ Failed to load from {p}: {str(e)[:100]}")
            last_err = e
            continue
    
    if tok is None:
        raise RuntimeError(f"Failed to load tokenizer from any of {tried_paths}: {last_err}")
    
    # 토크나이저 설정
    tok.padding_side = "left"
    tok.model_max_length = max_length
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
        print(f"Set pad_token_id to: {tok.pad_token_id}")
    
    return tok


def safe_decode(tokenizer, token_ids: List[int]) -> str:
    """안전한 토큰 디코딩"""
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    except Exception as e:
        print(f"⚠️  Decode error: {e}")
        return ""

def load_model(model_path: str, adapter_path: Optional[str], dtype, merge_and_unload: bool):
    """모델 로드 (PEFT 어댑터 지원)"""
    print(f"Loading model from: {model_path}")
    
    # case 1) base + adapter
    if adapter_path:
        print(f"Loading PEFT adapter from: {adapter_path}")
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype=dtype, 
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        if merge_and_unload:
            print("Merging and unloading PEFT weights...")
            model = model.merge_and_unload()
        model.eval()
        return model

    # case 2) adapter folder only (auto-detect PEFT)
    peft_files = ["adapter_config.json", "adapter_model.bin", "adapter_model.safetensors", "peft_config.json"]
    is_peft_folder = any(os.path.isfile(os.path.join(model_path, f)) for f in peft_files)
    
    if is_peft_folder:
        print("Detected PEFT adapter folder, loading with AutoPeftModel...")
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path, 
            device_map="auto", 
            torch_dtype=dtype, 
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        if merge_and_unload:
            print("Merging and unloading PEFT weights...")
            model = model.merge_and_unload()
        model.eval()
        return model

    # case 3) regular HF model
    print("Loading regular HuggingFace model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype=dtype, 
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.eval()
    return model

def extract_fields(example: Dict[str, Any], dataset_name: str) -> tuple:
    """데이터셋별 필드 추출"""
    config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["squad"])
    
    question = ""
    context = ""
    
    if isinstance(example, dict):
        question = example.get(config["question_key"], "")
        context = example.get(config["context_key"], "")
        if not context:
            # 추가 폴백
            context = example.get("text", "") or example.get("article", "") or example.get("passage", "")
    
    return str(question), str(context)


def extract_answers(example: Dict[str, Any], dataset_name: str) -> List[str]:
    """데이터셋별 정답 추출"""
    config = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["squad"])
    refs = []
    
    if isinstance(example, dict):
        answer_key = config["answer_key"]
        
        if answer_key in example:
            ans = example[answer_key]
            
            # SQuAD 스타일: {"text": [...], "answer_start": [...]}
            if isinstance(ans, dict) and "text" in ans:
                text_list = ans["text"]
                if isinstance(text_list, (list, tuple)):
                    refs = [str(x) for x in text_list]
                else:
                    refs = [str(text_list)]
            
            # 리스트 형태 답변
            elif isinstance(ans, list):
                if ans and isinstance(ans[0], dict):
                    # [{"text": "...", "answer_start": 123}, ...]
                    for a in ans:
                        if isinstance(a, dict):
                            text_data = a.get("text", a.get("answer", ""))
                            if isinstance(text_data, (list, tuple)):
                                refs.extend([str(x) for x in text_data])
                            elif text_data:
                                refs.append(str(text_data))
                else:
                    # ["answer1", "answer2", ...]
                    refs = [str(x) for x in ans]
            
            # 단일 문자열 답변
            else:
                refs = [str(ans)]
        
        # 추가 폴백 필드들
        for fallback in ["answers_text", "answers_texts", "answer", "ground_truth"]:
            if not refs and fallback in example:
                fallback_ans = example[fallback]
                if isinstance(fallback_ans, (list, tuple)):
                    refs = [str(x) for x in fallback_ans]
                else:
                    refs = [str(fallback_ans)]
                break
    
    return refs if refs else [""]


def analyze_results(results: List[Dict]) -> Dict[str, Any]:
    """결과 분석 및 통계 생성"""
    if not results:
        return {}
    
    # 기본 통계
    em_scores = [r["em"] for r in results]
    f1_scores = [r["f1"] for r in results]
    pred_lengths = [len(r["prediction"].split()) for r in results]
    
    analysis = {
        "total_examples": len(results),
        "em_mean": sum(em_scores) / len(em_scores) * 100,
        "f1_mean": sum(f1_scores) / len(f1_scores) * 100,
        "em_std": (sum((x - sum(em_scores)/len(em_scores))**2 for x in em_scores) / len(em_scores))**0.5 * 100,
        "f1_std": (sum((x - sum(f1_scores)/len(f1_scores))**2 for x in f1_scores) / len(f1_scores))**0.5 * 100,
        "avg_pred_length": sum(pred_lengths) / len(pred_lengths),
        "perfect_matches": sum(1 for x in em_scores if x == 1.0),
        "zero_scores": sum(1 for x in f1_scores if x == 0.0)
    }
    
    # 상위/하위 성능 예시
    sorted_by_f1 = sorted(results, key=lambda x: x["f1"], reverse=True)
    analysis["best_examples"] = sorted_by_f1[:3]
    analysis["worst_examples"] = sorted_by_f1[-3:]
    
    return analysis


@torch.no_grad()
def evaluate_batch(model, tokenizer, prompts: List[str], ref_lists: List[List[str]], 
                  gen_config: Dict[str, Any], device) -> List[Dict[str, Any]]:
    """배치 평가 함수"""
    max_length = gen_config.get("max_length", 1024)
    
    # 토크나이징
    enc = tokenizer(prompts, return_tensors="pt", padding="longest", 
                   truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_lens = attention_mask.sum(dim=1).cpu().tolist()
    
    # 생성
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=gen_config["max_new_tokens"],
        num_beams=gen_config["num_beams"],
        do_sample=gen_config["do_sample"],
        temperature=gen_config["temperature"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True,
        use_cache=True,
    )
    outputs = outputs.cpu().numpy().tolist()
    
    # 결과 처리
    batch_results = []
    for bi, out_ids in enumerate(outputs):
        plen = int(prompt_lens[bi])
        answer_ids = out_ids[plen:] if len(out_ids) > plen else []
        pred = safe_decode(tokenizer, answer_ids)
        refs = ref_lists[bi]
        metrics = metric_best(pred, refs)
        
        batch_results.append({
            "prediction": pred,
            "references": refs,
            "em": metrics["em"],
            "f1": metrics["f1"]
        })
    
    # 메모리 정리
    del input_ids, attention_mask, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return batch_results


# ---- main evaluation ----
def evaluate(
    model_path: str,
    tokenizer_path: Optional[str],
    dataset_name: str,
    split: str,
    template_name: str,
    device: str,
    batch_size: int,
    max_eval_samples: Optional[int],
    gen_max_new_tokens: int,
    num_beams: int,
    do_sample: bool,
    temperature: float,
    output_file: str,
    max_length: int = 1024,
    adapter_path: Optional[str] = None,
    merge_and_unload: bool = False,
    show_examples: int = 5,
):
    """메인 평가 함수"""
    dtype = torch.float16
    
    print("=" * 60)
    print("🚀 QA Model Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")
    if adapter_path:
        print(f"Adapter: {adapter_path}")
    print(f"Dataset: {dataset_name} ({split})")
    print(f"Template: {template_name}")
    print(f"Max samples: {max_eval_samples or 'All'}")
    print("=" * 60)
    
    # 토크나이저 로드
    print("\n📝 Loading tokenizer...")
    tokenizer = load_tokenizer(tokenizer_path, model_path, adapter_path, max_length)
    
    # 모델 로드
    print("\n🤖 Loading model...")
    model = load_model(model_path, adapter_path, dtype, merge_and_unload)
    
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(device)
    print(f"✓ Model loaded on device: {model_device}")
    
    # 데이터셋 로드
    print(f"\n📊 Loading dataset {dataset_name} split={split}...")
    try:
        raw = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        raise
    
    n_total = len(raw)
    if max_eval_samples is not None:
        raw = raw.select(range(min(max_eval_samples, n_total)))
    print(f"✓ Evaluating {len(raw)} examples (from {n_total} total)")
    
    # 생성 설정
    gen_config = {
        "max_length": max_length,
        "max_new_tokens": gen_max_new_tokens,
        "num_beams": num_beams if not do_sample else 1,
        "do_sample": do_sample,
        "temperature": temperature
    }
    
    # 평가 실행
    print(f"\n⚡ Starting evaluation (batch_size={batch_size})...")
    results = []
    em_sum = f1_sum = 0.0
    count = 0
    
    for i in tqdm(range(0, len(raw), batch_size), desc="Evaluating"):
        batch = [raw[k] for k in range(i, min(i + batch_size, len(raw)))]
        
        # 프롬프트 및 정답 준비
        prompts, ref_lists = [], []
        for ex in batch:
            question, context = extract_fields(ex, dataset_name)
            refs = extract_answers(ex, dataset_name)
            
            prompt = build_prompt(template_name, question, context)
            prompts.append(prompt)
            ref_lists.append(refs)
        
        # 배치 평가
        batch_results = evaluate_batch(model, tokenizer, prompts, ref_lists, gen_config, model_device)
        
        # 결과 수집
        for bi, batch_result in enumerate(batch_results):
            em_sum += batch_result["em"]
            f1_sum += batch_result["f1"]
            count += 1
            
            results.append({
                "index": i + bi,
                "question": prompts[bi].split('\n')[0][:100] + "...",  # 요약된 질문
                "prediction": batch_result["prediction"],
                "references": batch_result["references"],
                "em": batch_result["em"],
                "f1": batch_result["f1"]
            })
        
        # 중간 진행상황 출력
        if i > 0 and (i // batch_size) % 50 == 0:
            current_em = 100.0 * em_sum / count
            current_f1 = 100.0 * f1_sum / count
            print(f"Interim results - EM: {current_em:.2f}%, F1: {current_f1:.2f}%")
    
    # 최종 결과
    final_em = 100.0 * em_sum / max(1, count)
    final_f1 = 100.0 * f1_sum / max(1, count)
    
    print("\n" + "=" * 60)
    print("📈 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Examples evaluated: {count:,}")
    print(f"Exact Match (EM):   {final_em:.4f}%")
    print(f"F1 Score:           {final_f1:.4f}%")
    
    # 상세 분석
    analysis = analyze_results(results)
    if analysis:
        print(f"\nDetailed Analysis:")
        print(f"  EM std:            {analysis['em_std']:.4f}%")
        print(f"  F1 std:            {analysis['f1_std']:.4f}%")
        print(f"  Perfect matches:   {analysis['perfect_matches']:,} ({analysis['perfect_matches']/count*100:.1f}%)")
        print(f"  Zero F1 scores:    {analysis['zero_scores']:,} ({analysis['zero_scores']/count*100:.1f}%)")
        print(f"  Avg pred length:   {analysis['avg_pred_length']:.1f} tokens")
    
    # 예시 출력
    if show_examples > 0 and results:
        print(f"\n📝 Sample Results (top {show_examples}):")
        sorted_results = sorted(results, key=lambda x: x["f1"], reverse=True)
        for i, r in enumerate(sorted_results[:show_examples]):
            print(f"\n[{i+1}] F1: {r['f1']:.3f}, EM: {r['em']:.0f}")
            print(f"Q: {r['question']}")
            print(f"P: {r['prediction']}")
            print(f"A: {r['references'][0]}")
    
    # 결과 저장
    if output_file:
        print(f"\n💾 Saving results to {output_file}...")
        with open(output_file, "w", encoding="utf-8") as fout:
            # 메타데이터 저장
            metadata = {
                "model_path": model_path,
                "adapter_path": adapter_path,
                "dataset": dataset_name,
                "split": split,
                "template": template_name,
                "total_examples": count,
                "em_score": final_em,
                "f1_score": final_f1,
                "generation_config": gen_config
            }
            fout.write(json.dumps({"metadata": metadata}, ensure_ascii=False) + "\n")
            
            # 개별 결과 저장
            for r in results:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"✓ Results saved successfully!")
    
    print("=" * 60)
    return {"em": final_em, "f1": final_f1, "n": count, "analysis": analysis}


# ---- CLI ----
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLM QA Performance Evaluation Tool")
    
    # 모델 관련
    ap.add_argument("--model_path", type=str, required=True, 
                   help="베이스 HF 모델 폴더 또는 어댑터 폴더")
    ap.add_argument("--adapter_path", type=str, default=None, 
                   help="LoRA 어댑터 폴더(선택)")
    ap.add_argument("--tokenizer_path", type=str, default=None, 
                   help="토크나이저 경로(없으면 모델/어댑터에서 폴백)")
    ap.add_argument("--merge_and_unload", action="store_true",
                   help="PEFT 로드 시 LoRA 가중치를 베이스에 병합한 뒤 평가(속도↑, 메모리↑)")
    
    # 데이터셋 관련
    ap.add_argument("--dataset", type=str, default="squad", 
                   help="평가할 데이터셋 이름")
    ap.add_argument("--split", type=str, default="validation", 
                   help="데이터셋 분할 (train/validation/test)")
    ap.add_argument("--max_eval_samples", type=int, default=None, 
                   help="평가할 최대 샘플 수 (None=전체)")
    
    # 프롬프트 관련
    ap.add_argument("--template", type=str, default="squad_style", 
                   choices=list(TEMPLATES.keys()), help="프롬프트 템플릿")
    
    # 생성 관련
    ap.add_argument("--gen_max_new_tokens", type=int, default=128, 
                   help="생성할 최대 새 토큰 수")
    ap.add_argument("--num_beams", type=int, default=1, 
                   help="빔 서치 크기")
    ap.add_argument("--do_sample", action="store_true", 
                   help="샘플링 기반 생성 활성화")
    ap.add_argument("--temperature", type=float, default=1.0, 
                   help="생성 온도")
    
    # 시스템 관련
    ap.add_argument("--device", type=str, default="cuda:0", 
                   help="사용할 디바이스")
    ap.add_argument("--batch_size", type=int, default=4, 
                   help="배치 크기")
    ap.add_argument("--max_length", type=int, default=1024, 
                   help="최대 시퀀스 길이")
    
    # 출력 관련
    ap.add_argument("--output", type=str, default="qa_predictions.jsonl", 
                   help="결과 저장 파일명")
    ap.add_argument("--show_examples", type=int, default=5, 
                   help="출력할 예시 개수")
    
    args = ap.parse_args()
    
    # 평가 실행
    metrics = evaluate(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        dataset_name=args.dataset,
        split=args.split,
        template_name=args.template,
        device=args.device,
        batch_size=args.batch_size,
        max_eval_samples=args.max_eval_samples,
        gen_max_new_tokens=args.gen_max_new_tokens,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        temperature=args.temperature,
        output_file=args.output,
        max_length=args.max_length,
        adapter_path=args.adapter_path,
        merge_and_unload=args.merge_and_unload,
        show_examples=args.show_examples,
    )
    
    print(f"\n Evaluation completed successfully!")
    print(f"Final metrics: EM={metrics['em']:.4f}%, F1={metrics['f1']:.4f}%")