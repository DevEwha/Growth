"""
Progressive LLM Loading
캐시 정리: sudo sync; echo 1 | sudo tee /proc/sys/vm/drop_caches
"""

from glob import glob
import os
import torch
import json
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from typing import List
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, pipeline
from transformers.modeling_utils import load_sharded_checkpoint
from peft import LoraConfig, get_peft_model, PeftModel

from model_utils import *


@dataclass
class Config:
    """설정 클래스"""
    base_dir: str = "./models/pruned_model_with_adapters"
    device: str = "cuda:0"
    max_length: int = 150
    temperature: float = 0.7
    seqlen: int = 4096

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
    def a_adapter_config_path(self) -> str:
        return os.path.join(self.adapter_dir, "A_lora", "stageA", "adapter_config.json")
    
    @property
    def a_adapter_pt_path(self) -> str:
        return os.path.join(self.adapter_dir, "A_lora", "stageA.pt")
    
    @property
    def ab_adapter_config_path(self) -> str:
        return os.path.join(self.adapter_dir, "AB_lora", "stageAB", "adapter_config.json")
    
    @property
    def ab_adapter_pt_path(self) -> str:
        return os.path.join(self.adapter_dir, "AB_lora", "stageAB.pt")
    
    @property
    def abc_adapter_config_path(self) -> str:
        return os.path.join(self.adapter_dir, "ABC_lora", "stageABC", "adapter_config.json")
    
    @property
    def abc_adapter_pt_path(self) -> str:
        return os.path.join(self.adapter_dir, "ABC_lora", "stageABC.pt")


def print_header(text: str, char: str = "="):
    """깔끔한 헤더 출력"""
    width = 80
    print(f"\n{char * width}")
    print(f"{text.center(width)}")
    print(f"{char * width}\n")


def print_step(step_num: int, text: str):
    """단계별 진행 상황 출력"""
    print(f"\n{'▶'*3} STAGE {step_num}: {text}")


def print_info(text: str):
    """정보 메시지 출력"""
    print(f"  ℹ️  {text}")


def print_success(text: str):
    """성공 메시지 출력"""
    print(f"  ✅ {text}")


def log_mem(tag=""):
    """메모리 사용량 출력"""
    alloc = torch.cuda.memory_allocated() / 1e9
    reserv = torch.cuda.memory_reserved() / 1e9
    print(f"  💾 {tag}: GPU 메모리 사용량 = {alloc:.2f}GB / {reserv:.2f}GB")


def free_cuda(*objs):
    """참조 해제 및 캐시 비우기"""
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def generate_sample(model, tokenizer, stage_name: str, prompt: str, max_new_tokens: int = 50):
    """샘플 텍스트 생성"""
    try:
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)
            
            print_info("텍스트 생성 중...")
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\n{'─'*80}")
            print(f"📝 [{stage_name}] 생성된 텍스트:")
            print(f"{'─'*80}")
            print(f"{generated_text}")
            print(f"{'─'*80}\n")
            
    except Exception as e:
        print(f"  ❌ 생성 실패 ({stage_name}): {e}")


def main():
    print_header("🚀 Progressive LLM Loading", "=")
    
    # 사용자 입력 받기
    print("\n" + "="*80)
    user_prompt = input("🎤 프롬프트를 입력하세요: ").strip()
    if not user_prompt:
        user_prompt = "Paris is"
        print_info(f"기본 프롬프트 사용: {user_prompt}")
    print("\n" + "="*80)
    
    config = Config()
    print_info(f"디바이스: {config.device}")
    print_info(f"베이스 디렉토리: {config.base_dir}")
    
    device = config.device if torch.cuda.is_available() else "cpu"
    log_mem("초기 상태")
    
    # ==================== STAGE 1: 모델 A 로딩 ====================
    print_step(1, "모델 A 로딩 (Pruned Model)")
    
    print_info("토크나이저 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(str(config.a_dir))
    print_success("토크나이저 로딩 완료")
    
    print_info("모델 A 로딩 중...")
    model = AutoModelForCausalLM.from_pretrained(
        config.a_dir,
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )
    model.config.use_cache = False
    model.to(device)
    print_success("모델 A 로딩 완료")
    log_mem("모델 A 로딩 후")
    
    print_info("PassLayer 적용 중...")
    manifest_path = os.path.join(config.a_dir, "manifest.json")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    removed = find_removed_layers(manifest)
    model = install_pass_layers(
        model,
        removed_indices=removed,
        get_layer_container=get_layer_container,
        get_layer_device=lambda layer: torch.device("cuda:0"),
        default_device=torch.device("cuda:0"),
    )
    print_success("PassLayer 적용 완료")
    
    print_info("Stage A 어댑터 적용 중...")
    adapter_config = json.load(open(config.a_adapter_config_path, "r", encoding="utf-8"))
    lora_config = LoraConfig(
        task_type=adapter_config.get("task_type", "CAUSAL_LM"),
        r=adapter_config.get("r", 16),
        lora_alpha=adapter_config.get("lora_alpha", 32),
        target_modules=adapter_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        lora_dropout=adapter_config.get("lora_dropout", 0.05),
        bias=adapter_config.get("bias", "none"),
        inference_mode=False
    )
    model = get_peft_model(model, lora_config)
    
    checkpoint = torch.load(config.a_adapter_pt_path, map_location=model.device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint
    
    fixed_state_dict = fix_parameter_keys(state_dict, model)
    missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)
    
    lora_loaded = len([k for k in fixed_state_dict.keys() if 'lora_' in k])
    print_success(f"Stage A 어댑터 적용 완료 (LoRA 파라미터: {lora_loaded}개)")
    log_mem("어댑터 A 적용 후")
    
    generate_sample(model, tokenizer, "Stage 1: Model A", user_prompt, max_new_tokens=50)
    
    # ==================== STAGE 2: 모델 A+B 로딩 ====================
    print_step(2, "모델 B 복구 (A + B)")
    
    print_info("prune_log.json 읽는 중...")
    path = os.path.join(config.a_dir, "prune_log.json")
    with open(path, "r", encoding="utf-8") as f:
        log = json.load(f)
    B_idx, C_idx = log["split"]["B"], log["split"]["C"]
    print_info(f"레이어 인덱스 - B: {len(B_idx)}개, C: {len(C_idx)}개")
    
    print_info("B 레이어 복구 중...")
    rehydrate_layers(model, config.b_dir, B_idx)
    print_success("B 레이어 복구 완료")
    
    print_info("Stage AB 어댑터 적용 중...")
    adapter_config = json.load(open(config.ab_adapter_config_path, "r", encoding="utf-8"))
    lora_config = LoraConfig(
        task_type=adapter_config.get("task_type", "CAUSAL_LM"),
        r=adapter_config.get("r", 16),
        lora_alpha=adapter_config.get("lora_alpha", 32),
        target_modules=adapter_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        lora_dropout=adapter_config.get("lora_dropout", 0.05),
        bias=adapter_config.get("bias", "none"),
        inference_mode=False
    )
    
    adapter_name = "adapter_AB"
    model.add_adapter(peft_config=lora_config, adapter_name=adapter_name)
    model.set_adapter(adapter_name)
    
    checkpoint = torch.load(config.ab_adapter_pt_path, map_location=model.device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint
    
    fixed_state_dict = fix_parameter_keys(state_dict, model)
    missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)
    print_success("Stage AB 어댑터 적용 완료")
    log_mem("어댑터 AB 적용 후")
    
    generate_sample(model, tokenizer, "Stage 2: Model A+B", user_prompt, max_new_tokens=50)
    
    # ==================== STAGE 3: 전체 모델 (A+B+C) ====================
    print_step(3, "모델 C 복구 (A + B + C - 전체 모델)")
    
    print_info("C 레이어 복구 중...")
    rehydrate_layers(model, config.c_dir, C_idx)
    print_success("C 레이어 복구 완료")
    log_mem("전체 모델 복구 후")
    
    generate_sample(model, tokenizer, "Stage 3: Full Model (A+B+C)", user_prompt, max_new_tokens=50)
    
    print_header("✨ 완료", "=")


if __name__ == "__main__":
    main()
