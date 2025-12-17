from glob import glob
import os
import torch
import json
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from peft import LoraConfig, get_peft_model
from model_utils import *
from datasets import load_dataset
from evaluate import load
from tqdm import tqdm
import csv
import time
from contextlib import contextmanager
from log import *
import re


logger = setup_logger(name="runner", log_dir="./logs")

SAMPLE_COUNT = 900


@contextmanager
def wall_timer(label: str = ""):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception(f"[WALL]{' ' + label if label else ''} 실행 중 예외 발생")
        raise
    finally:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        dt = time.perf_counter() - t0
        logger.info(f"[TIME][WALL]{' ' + label if label else ''}: {dt*1000:.2f} ms")


@contextmanager
def gpu_timer(label: str = ""):
    if not torch.cuda.is_available():
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    try:
        yield
    except Exception:
        logger.exception(f"[GPU]{' ' + label if label else ''} 실행 중 예외 발생")
        raise
    finally:
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        logger.info(f"[TIME][GPU]{' ' + label if label else ''}: {ms:.2f} ms")


@dataclass
class Config:
    base_dir: str = "/home/devewha/new_model"
    device: str = "cuda:0"
    
    @property
    def a_dir(self) -> str:
        return f"{self.base_dir}/A"
    
    @property
    def b_dir(self) -> str:
        return f"{self.base_dir}/bundles/B"
    
    @property
    def c_dir(self) -> str:
        return f"{self.base_dir}/bundles/C"
    
    @property
    def adapter_dir(self) -> str:
        return f"{self.base_dir}/adapters"
    
    @property
    def abc_adapter_config_path(self) -> str:
        return os.path.join(self.adapter_dir, "ABC_lora", "stageABC", "adapter_config.json")
    
    @property
    def abc_adapter_pt_path(self) -> str:
        return os.path.join(self.adapter_dir, "ABC_lora", "stageABC.pt")


def log_mem(tag=""):
    if torch.cuda.is_available():
        print(f"[MEM] {tag} | alloc={torch.cuda.memory_allocated()/1e9:.2f}G, "
              f"reserv={torch.cuda.memory_reserved()/1e9:.2f}G")


def free_cuda(*objs):
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def normalize_device(device_str: str) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    want_idx = 0
    if ":" in device_str:
        try:
            want_idx = int(device_str.split(":")[1])
        except:
            want_idx = 0
    n = torch.cuda.device_count()
    if want_idx >= n:
        logger.warning(f"[Device] requested {device_str} but only {n} visible device(s). Using cuda:0")
        want_idx = 0
    return f"cuda:{want_idx}"


def _has_meta_tensors(module: torch.nn.Module) -> bool:
    """모델에 meta tensor가 남아있는지 확인"""
    for n, p in module.named_parameters(recurse=True):
        if getattr(p, "is_meta", False) or p.device.type == "meta":
            logger.error(f"[META] parameter on meta: {n}")
            return True
    for n, b in module.named_buffers(recurse=True):
        if getattr(b, "is_meta", False) or b.device.type == "meta":
            logger.error(f"[META] buffer on meta: {n}")
            return True
    return False


def evaluate_triviaqa_fixed(model, tokenizer, num_samples=1000):
    logger.info("TriviaQA 데이터셋 로드 중...")
    dataset = load_dataset("trivia_qa", "rc", split="validation")
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    logger.info(f"평가할 샘플 수: {len(dataset)}")
    
    squad_metric = load("squad")
    predictions, references = [], []
    model.eval()
    
    skipped = 0
    
    for idx, example in enumerate(tqdm(dataset, desc="Generating predictions")):
        try:
            question = example["question"]
            answer_data = example["answer"]
            
            if isinstance(answer_data, dict):
                answer_text = answer_data.get("value", "")
                answer_aliases = answer_data.get("aliases", [])
                all_answers = [answer_text] + (answer_aliases or [])
            else:
                answer_text = str(answer_data)
                all_answers = [answer_text]
            
            # 원본 context 사용
            context = ""
            if "entity_pages" in example and example["entity_pages"]:
                entity_pages = example["entity_pages"]
                if isinstance(entity_pages, dict) and "wiki_context" in entity_pages:
                    wiki_context = entity_pages["wiki_context"]
                    if isinstance(wiki_context, list):
                        context = " ".join(str(x) for x in wiki_context[:3] if x)
                    else:
                        context = str(wiki_context)
                elif isinstance(entity_pages, list) and len(entity_pages) > 0:
                    first_page = entity_pages[0]
                    if isinstance(first_page, dict):
                        wiki_context = first_page.get("wiki_context", "")
                        if isinstance(wiki_context, list):
                            context = " ".join(str(x) for x in wiki_context[:3] if x)
                        else:
                            context = str(wiki_context)
            
            if not context and "search_results" in example:
                search_results = example["search_results"]
                if isinstance(search_results, dict) and "search_context" in search_results:
                    search_context = search_results["search_context"]
                    if isinstance(search_context, list):
                        context = " ".join(str(x) for x in search_context[:3] if x)
                    else:
                        context = str(search_context)
                elif isinstance(search_results, list) and len(search_results) > 0:
                    first_result = search_results[0]
                    if isinstance(first_result, dict):
                        search_context = first_result.get("search_context", "")
                        if isinstance(search_context, list):
                            context = " ".join(str(x) for x in search_context[:3] if x)
                        else:
                            context = str(search_context)
            
            context = context.strip()
            if not context:
                skipped += 1
                continue
            
            context = context[:1500]
            example_id = str(idx)
            
            # 단순한 프롬프트
            prompt = f"{context}\n\nQ: {question}\nA:"
            
            with torch.no_grad():
                inputs = tokenizer(
                    prompt, 
                    return_tensors="pt",
                    truncation=True, 
                    max_length=1024
                )
                input_ids = inputs["input_ids"].to(model.device)
                attention_mask = inputs.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(model.device)
                
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=1.0,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                prompt_dec = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                pred_answer = generated_text[len(prompt_dec):].strip()
                
                if '\n' in pred_answer:
                    pred_answer = pred_answer.split('\n')[0].strip()
                pred_answer = pred_answer.strip('.,;:"\' ')
                
                if len(pred_answer.split()) > 10:
                    pred_answer = " ".join(pred_answer.split()[:10])
            
            if not pred_answer:
                pred_answer = "no answer"
            
            predictions.append({
                "id": example_id,
                "prediction_text": pred_answer
            })
            
            references.append({
                "id": example_id,
                "answers": {
                    "text": all_answers,
                    "answer_start": [0] * len(all_answers)
                }
            })
            
            if len(predictions) <= 10:
                logger.info(f"\n{'='*80}")
                logger.info(f"[Sample {len(predictions)}]")
                logger.info(f"Q: {question}")
                logger.info(f"Predicted: {pred_answer}")
                logger.info(f"True answers: {all_answers[:3]}")
                logger.info(f"{'='*80}")
        
        except Exception as e:
            logger.warning(f"샘플 {idx} 처리 중 에러: {e}")
            skipped += 1
            continue
    
    logger.info(f"스킵: {skipped}, 평가: {len(predictions)}")
    
    if len(predictions) == 0:
        return {"exact_match": 0.0, "f1": 0.0}
    
    results = squad_metric.compute(predictions=predictions, references=references)
    return results
def main():
    logger.info("=" * 80)
    logger.info("Stage ABC (전체 모델 복구 - Adapter 없음) 평가 시작")
    logger.info("=" * 80)
    log_mem("start")
    
    config = Config()
    logger.info(f"환경 설정: device={config.device}, base_dir={config.base_dir}")
    
    safe_device = normalize_device(config.device)
    logger.info(f"실제 디바이스(정규화): {safe_device}")
    
    # 토크나이저 로드
    with wall_timer("load_tokenizer"), gpu_timer("load_tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(str(config.a_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    log_mem("after_tokenizer")
    
    # 모델 로드
    with wall_timer("load_model"), gpu_timer("load_model"):
        model = AutoModelForCausalLM.from_pretrained(
            config.a_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    log_mem("after_model_load")
    
    # Meta tensor 확인
    if _has_meta_tensors(model):
        raise RuntimeError("Model still has meta tensors after from_pretrained.")
    
    # PassLayer 적용
    with wall_timer("PassLayer적용"), gpu_timer("PassLayer적용"):
        logger.info("PassLayer 적용")
        manifest_path = os.path.join(config.a_dir, "manifest.json")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        removed = find_removed_layers(manifest)
        logger.info(f"제거된 레이어: {removed}")
        model = install_new_pass_layers(
            model,
            removed_indices=removed,
            get_layer_container=get_layer_container,
            get_layer_device=lambda layer: model.device,
            default_device=model.device,
        )
    log_mem("after_passlayer")
    
    # Prune log 읽기
    with wall_timer("read_prune_log"):
        path = os.path.join(config.a_dir, "prune_log.json")
        logger.info(f"prune_log 읽기: {path}")
        with open(path, "r", encoding="utf-8") as f:
            log = json.load(f)
        B_idx, C_idx = log["split"]["B"], log["split"]["C"]
        logger.info(f"인덱스: B={len(B_idx)}, C={len(C_idx)}")
    
    # B 모델 복구
    with wall_timer("B 모델 복구"), gpu_timer("B 모델 복구"):
        logger.info("B 모델 복구 시작")
        rehydrate_layers(model, config.b_dir, B_idx)
        logger.info(f"B 레이어 {len(B_idx)}개 복구 완료")
    log_mem("after_B_rehydrate")
    
    # C 모델 복구 (전체 모델 복구 완료)
    with wall_timer("C 모델 복구"), gpu_timer("C 모델 복구"):
        logger.info("C 모델 복구 시작")
        rehydrate_layers(model, config.c_dir, C_idx)
        logger.info(f"C 레이어 {len(C_idx)}개 복구 완료")
        total_restored = len(B_idx) + len(C_idx)
        logger.info(f"✓ 전체 모델 복구 완료 (B:{len(B_idx)} + C:{len(C_idx)} = {total_restored}개 레이어)")
    log_mem("after_full_rehydrate")
    
    # ★ Adapter 적용 없음 - 순수 전체 복구 모델로 평가
    logger.info("=" * 80)
    logger.info("Adapter 없이 순수 전체 복구 모델로 평가")
    logger.info("=" * 80)
    
    # TriviaQA 평가
    logger.info("\n" + "=" * 80)
    logger.info("TriviaQA 평가 시작 (전체 모델 - Adapter 없음)")
    logger.info("=" * 80)
    results = evaluate_triviaqa_fixed(model, tokenizer, num_samples=SAMPLE_COUNT)
    logger.info(f"평가 결과: {results}")
    
    # 결과 저장
    output_file = f"full_model_no_adapter_triviaqa_{SAMPLE_COUNT}.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for metric_name, value in results.items():
            writer.writerow([metric_name, value])
    
    logger.info(f"Evaluation results saved to {output_file}")
    print(f"\n{'='*80}")
    print(f"전체 모델 복구 (Adapter 없음) TriviaQA Evaluation Results:")
    print(f"  Total Layers Restored: {len(B_idx) + len(C_idx)}")
    print(f"  Configuration: Full Model (A+B+C) without LoRA")
    for metric_name, value in results.items():
        print(f"  {metric_name}: {value:.2f}")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*80}\n")
    
    # 메모리 정리
    free_cuda(model, tokenizer)
    log_mem("after_cleanup")


if __name__ == "__main__":
    main()
