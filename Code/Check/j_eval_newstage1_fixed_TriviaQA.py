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
    base_dir: str = "/acpl-ssd20/nadam_model_0916"
    device: str = "cuda:0"
    
    @property
    def a_dir(self) -> str:
        return f"{self.base_dir}/A"
    
    @property
    def adapter_dir(self) -> str:
        return f"{self.base_dir}/adapters"
    
    @property
    def a_adapter_config_path(self) -> str:
        return os.path.join(self.adapter_dir, "A_lora", "stageA", "adapter_config.json")
    
    @property
    def a_adapter_pt_path(self) -> str:
        return os.path.join(self.adapter_dir, "A_lora", "stageA.pt")


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


# -----------------------------
# Device normalization helpers
# -----------------------------
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


# -----------------------------
# Context building & heuristics
# -----------------------------
def sent_tokenize(text: str):
    return re.split(r'(?<=[\.\?\!])\s+', text)


def build_context(example, question, aliases, max_chars=2000, top_k=20):
    pieces = []

    # 1) entity_pages
    ep = example.get("entity_pages", None)
    if isinstance(ep, dict):
        wk = ep.get("wiki_context", [])
        if isinstance(wk, list): 
            pieces += [str(x) for x in wk if x]
        elif isinstance(wk, str): 
            pieces.append(wk)
    elif isinstance(ep, list):
        for page in ep:
            if isinstance(page, dict):
                wk = page.get("wiki_context", [])
                if isinstance(wk, list): 
                    pieces += [str(x) for x in wk if x]
                elif isinstance(wk, str): 
                    pieces.append(wk)

    # 2) search_results
    sr = example.get("search_results", None)
    if isinstance(sr, dict):
        sc = sr.get("search_context", [])
        if isinstance(sc, list): 
            pieces += [str(x) for x in sc if x]
        elif isinstance(sc, str): 
            pieces.append(sc)
    elif isinstance(sr, list):
        for res in sr:
            if isinstance(res, dict):
                sc = res.get("search_context", "")
                if isinstance(sc, list): 
                    pieces += [str(x) for x in sc if x]
                elif isinstance(sc, str): 
                    pieces.append(sc)

    # Join and split into sentences
    sents = []
    for chunk in pieces:
        sents += sent_tokenize(chunk)

    q_words = set(re.findall(r"[A-Za-z][A-Za-z\-']+", str(question).lower()))
    alias_set = {str(a).lower() for a in aliases if isinstance(a, str)}

    def score(sent):
        s = sent.lower()
        sc = sum(w in s for w in q_words)
        sc += 3 * sum(a in s for a in alias_set)
        # 노이즈 감점
        if "imdb" in s or "there was an error" in s:
            sc -= 2
        return sc

    ranked = sorted(sents, key=score, reverse=True)
    new_ctx, cur = [], 0
    for s in ranked[:max(top_k, 20)]:
        s = s.strip()
        if not s:
            continue
        if cur + len(s) + 1 > max_chars:
            break
        new_ctx.append(s)
        cur += len(s) + 1

    context = " ".join(new_ctx).strip()
    return context


def alias_in_context(context, aliases):
    c = context.lower()
    for a in aliases:
        s = str(a).strip().lower()
        if s and s in c:
            return True
    return False


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


# -----------------------------
# Evaluation
# -----------------------------
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
            
            # ★ 원본 context 그대로 사용 (build_context 대신)
            context = ""
            if "entity_pages" in example and example["entity_pages"]:
                entity_pages = example["entity_pages"]
                if isinstance(entity_pages, dict) and "wiki_context" in entity_pages:
                    wiki_context = entity_pages["wiki_context"]
                    if isinstance(wiki_context, list):
                        context = " ".join(str(x) for x in wiki_context[:3] if x)  # 처음 3개만
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
            
            # Context 길이 제한
            context = context[:1500]
            
            example_id = str(idx)
            
            # ★ 매우 단순한 프롬프트 (Base 모델용)
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
                    max_new_tokens=10,  # ★ 15 → 10
                    do_sample=False,
                    temperature=1.0,    # ★ 명시적 설정
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                prompt_dec = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                pred_answer = generated_text[len(prompt_dec):].strip()
                
                # 후처리
                if '\n' in pred_answer:
                    pred_answer = pred_answer.split('\n')[0].strip()
                pred_answer = pred_answer.strip('.,;:"\' ')
                
                # ★ 너무 긴 답변 자르기 (보통 정답은 짧음)
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
            
            # 첫 10개 샘플 로깅
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
    logger.info("실행 시작")
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

    # ★ 모델 로드 개선: device_map 사용, low_cpu_mem_usage=True
    with wall_timer("load_model"), gpu_timer("load_model"):
        model = AutoModelForCausalLM.from_pretrained(
            config.a_dir,
            torch_dtype=torch.float16,
            device_map="auto",           # ★ GPU로 직접 로드
            low_cpu_mem_usage=True,      # ★ 메모리 효율성 향상
            trust_remote_code=True
        )
    log_mem("after_model_load")

    # Meta tensor 잔존 확인
    if _has_meta_tensors(model):
        raise RuntimeError(
            "Model still has meta tensors after from_pretrained. "
            "Check weight shards/index files under config.a_dir."
        )

    # device_map="auto"를 사용하므로 수동 .to() 호출 불필요

    # PassLayer 적용
    with wall_timer("PassLayer적용"), gpu_timer("PassLayer적용"):
        logger.info("PassLayer 적용")
        manifest_path = os.path.join(config.a_dir, "manifest.json")
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        removed = find_removed_layers(manifest)
        model = install_new_pass_layers(
            model,
            removed_indices=removed,
            get_layer_container=get_layer_container,
            get_layer_device=lambda layer: model.device,  # 모델의 현재 device 사용
            default_device=model.device,
        )
    log_mem("after_passlayer")

    # A Adapter 적용 개선: inference_mode=True
    with wall_timer("A Adapter 적용"), gpu_timer("A Adapter 적용"):
        logger.info("A 어댑터 적용")
        adapter_config = json.load(open(config.a_adapter_config_path, "r", encoding="utf-8"))
        lora_config = LoraConfig(
            task_type=adapter_config.get("task_type", "CAUSAL_LM"),
            r=adapter_config.get("r", 16),
            lora_alpha=adapter_config.get("lora_alpha", 32),
            target_modules=adapter_config.get("target_modules", 
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            lora_dropout=adapter_config.get("lora_dropout", 0.05),
            bias=adapter_config.get("bias", "none"),
            inference_mode=True  # 평가 시 True로 설정
        )
        model = get_peft_model(model, lora_config, adapter_name="adapter_A")
        
        # LoRA 가중치 로드
        checkpoint = torch.load(config.a_adapter_pt_path, map_location=model.device)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        else:
            state_dict = checkpoint

        fixed_state_dict = fix_parameter_keys(state_dict, model)
        missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)
        model.set_adapter("adapter_A")
        logger.info(f"LoRA 로드 완료 - Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
    log_mem("after_lora_load")

    # TriviaQA 평가
    results = evaluate_triviaqa_fixed(model, tokenizer, num_samples=SAMPLE_COUNT)
    logger.info(f"평가 결과: {results}")

    # 결과 저장
    output_file = f"stage1_triviaqa_eval_{SAMPLE_COUNT}.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for metric_name, value in results.items():
            writer.writerow([metric_name, value])

    logger.info(f"Evaluation results saved to {output_file}")
    print(f"\n{'='*80}")
    print(f"TriviaQA Evaluation Results:")
    for metric_name, value in results.items():
        print(f"  {metric_name}: {value}")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*80}\n")

    # 메모리 정리
    free_cuda(model, tokenizer)
    log_mem("after_cleanup")


if __name__ == "__main__":
    main()
