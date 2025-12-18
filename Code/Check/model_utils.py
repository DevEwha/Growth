import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from typing import Dict
from typing import Optional, List, Any
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import os

def safe_to_device(model: torch.nn.Module, device: str):
    """메타 디바이스에서 안전하게 디바이스 이동"""
    if hasattr(model, 'to_empty'):
        return model.to_empty(device=device)
    else:
        return model.to(device)

def load_model_config(model_path: Path):
    """모델 설정 로딩"""
    config_dir = model_path / "original_config"
    if not config_dir.exists():
        raise FileNotFoundError(f"설정 폴더가 없습니다: {config_dir}")
    return AutoConfig.from_pretrained(str(config_dir))

def load_model_tokenizer(model_path: Path):
    """토크나이저 로딩"""
    config_dir = model_path / "original_config"
    return AutoTokenizer.from_pretrained(str(config_dir))

def load_manifest(model_path: Path):
    """manifest.json 로딩"""
    import json
    manifest_path = model_path / "manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    p_layers = set(manifest['blocks']['P']['layers'])
    r1_layers = set(manifest['blocks']['R1']['layers'])
    r2_layers = set(manifest['blocks']['R2']['layers'])
    
    return manifest, p_layers, r1_layers, r2_layers

def load_safetensors_file(model_path: Path, filename: str, target_device: str = "cpu") -> Dict[str, torch.Tensor]:
    """SafeTensors 파일 로딩"""
    file_path = model_path / filename
    if not file_path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {file_path}")
    
    state_dict = load_file(str(file_path))
    
    if target_device != "cpu":
        state_dict = {k: v.to(target_device) for k, v in state_dict.items()}
    
    return state_dict

def create_zero_initialized_model(config, use_meta_device_first=True):
    """0으로 초기화된 모델 생성"""
    use_meta_device = False
    
    if use_meta_device_first:
        try:
            with torch.device("meta"):
                model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
            use_meta_device = True
        except Exception:
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
            use_meta_device = False
    else:
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        use_meta_device = False
    
    # 0으로 초기화
    if not use_meta_device:
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.zero_()
    
    return model, use_meta_device

def apply_state_dict_safe(model, state_dict, use_meta_device=False):
    """안전한 state_dict 적용"""
    if use_meta_device:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
    else:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    return missing_keys, unexpected_keys

def simple_inference_test(model, tokenizer, device="cpu", text="Hello, this is a test."):
    """간단한 추론 테스트"""
    if model is None:
        return None, None
    
    model_device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model_device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        generated = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text, outputs

#나경이 코드에서 가져온 함수
def find_removed_layers(manifest: dict) -> Optional[List[int]]:
    """매니페스트에서 제거된 레이어 찾기""" # simdrop 스키마
    removed = manifest.get("simdrop", {}).get("removed_layers")
    if removed:
        return removed
            
    # top-level 스키마
    removed = manifest.get("removed_layers")
    if removed:
        return removed
            
    # stages 스키마
    stages = manifest.get("stages", {})
    if stages:
        A_drop = stages.get("A", {}).get("dropped_layers", [])
        B_rem = stages.get("B", {}).get("removed_layers", [])
        C_rem = stages.get("C", {}).get("removed_layers", [])
        return A_drop or sorted(set(B_rem + C_rem))
            
    return None


# utils/pass_layers.py
from typing import List, Callable, Optional
import torch.nn as nn

def install_pass_layers(
    model,
    removed_indices: List[int],
    get_layer_container: Callable,     # fn(model) -> list-like container or None
    get_layer_device: Callable,        # fn(module) -> device or None
    default_device: Optional[str] = None,
):
    # PassLayer 구현 선택
    try:
        from lib.identity import LlamaPassLayer as _Inner
        class PassWrapper(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.inner = _Inner(hidden_size)
            def forward(self, hidden_states, *args, **kwargs):
                out = self.inner(hidden_states, *args, **kwargs)
                return out[0] if isinstance(out, tuple) else out
        print("[reapply] using project LlamaPassLayer")
    except Exception:
        class PassWrapper(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
            def forward(self, x, *args, **kwargs):
                return x
        print("[reapply] using SafePassLayer")

    # 레이어 컨테이너
    layer_container = get_layer_container(model)
    if layer_container is None:
        print("[reapply] cannot locate layers -> skip")
        return model

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("model.config.hidden_size가 필요합니다.")

    applied = []
    for i in removed_indices:
        if 0 <= i < len(layer_container):
            dev = get_layer_device(layer_container[i]) or default_device
            wrapper = PassWrapper(hidden_size)
            if dev is not None:
                wrapper = wrapper.to(dev)
            layer_container[i] = wrapper
            applied.append(i)

    print(f"[reapply] installed PassLayer on: {applied}")
    return model

def install_new_pass_layers(
    model,
    removed_indices: List[int],
    get_layer_container: Callable,
    get_layer_device: Callable,
    default_device: Optional[str] = None,
):
    # PassLayer 구현 선택
    try:
        from lib.identity import LlamaPassLayer as _Inner
        class PassWrapper(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.inner = _Inner(hidden_size)
            def forward(self, hidden_states, *args, **kwargs):
                out = self.inner(hidden_states, *args, **kwargs)
                # 이미 tuple이면 그대로 반환
                if isinstance(out, tuple):
                    return out
                # 아니면 tuple로 변환 (cache 지원)
                use_cache = kwargs.get('use_cache', False)
                output_attentions = kwargs.get('output_attentions', False)
                
                outputs = (out,)
                if use_cache:
                    outputs = outputs + (None,)  # past_key_value placeholder
                if output_attentions:
                    outputs = outputs + (None,)  # attention_weights placeholder
                return outputs
        print("[reapply] using project LlamaPassLayer")
    except Exception:
        class PassWrapper(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
            def forward(self, hidden_states, *args, **kwargs):
                # Llama 레이어와 호환되는 출력 형식
                use_cache = kwargs.get('use_cache', False)
                output_attentions = kwargs.get('output_attentions', False)
                
                # (hidden_states,) 튜플 시작
                outputs = (hidden_states,)
                
                # use_cache=True면 KV cache placeholder 추가
                if use_cache:
                    outputs = outputs + (None,)
                
                # output_attentions=True면 attention placeholder 추가
                if output_attentions:
                    outputs = outputs + (None,)
                
                return outputs
        print("[reapply] using SafePassLayer")

    # 레이어 컨테이너
    layer_container = get_layer_container(model)
    if layer_container is None:
        print("[reapply] cannot locate layers -> skip")
        return model

    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("model.config.hidden_size가 필요합니다.")

    applied = []
    for i in removed_indices:
        if 0 <= i < len(layer_container):
            dev = get_layer_device(layer_container[i]) or default_device
            wrapper = PassWrapper(hidden_size)
            if dev is not None:
                wrapper = wrapper.to(dev)
            layer_container[i] = wrapper
            applied.append(i)

    print(f"[reapply] installed PassLayer on: {applied}")
    return model

CANDIDATE_LAYER_PATHS = [
        "model.layers",
        "model.decoder.layers",
        "model.model.layers",
        "model.model.decoder.layers",
        "base_model.model.layers",
        "base_model.model.decoder.layers",
        "base_model.model.model.layers",
        "base_model.model.model.decoder.layers",
]

def get_layer_container(model) -> Optional[object]:
    for path in CANDIDATE_LAYER_PATHS:
        try:
            container = model
            for attr in path.split("."):
                container = getattr(container, attr)
            if hasattr(container, "__len__") and hasattr(container, "__getitem__"):
                return container
        except AttributeError:
            continue
    return None
    

def rehydrate_layers(model, bundle_dir: str, indices: List[int]):
    
    
    layers = get_layer_container(model)
    dtype = next(model.parameters()).dtype
    tgt = next(model.parameters()).device   # ← 단일 디바이스로 고정


    for i in indices:
        new_layer = LlamaDecoderLayer(model.config, layer_idx=i).to(device=tgt, dtype=dtype)
        f = os.path.join(bundle_dir, f"layer_{i:03d}.safetensors")
        if not os.path.isfile(f):
            raise FileNotFoundError(f"bundle miss: {f}")
        sd = load_file(f)
        sd = {k: v.to(device=tgt, dtype=dtype) for k, v in sd.items()}
        try:
            new_layer.load_state_dict(sd, strict=True)
        except RuntimeError as e:
            print(f"[warn] strict load failed for {i}: {e} -> non-strict")
            new_layer.load_state_dict(sd, strict=False)
        layers[i] = new_layer
        print(f"[rehydrate] layer {i} restored on {tgt}")






# peft 관련 유틸

def fix_parameter_keys(
    state_dict: Dict[str, Any],
    model: Optional[object] = None,  # PeftModel이 아니어도 동작하도록 완화
) -> Dict[str, Any]:
    """
    LoRA state_dict의 키를 현재 어댑터 이름에 맞게 리맵한다.
    model이 PeftModel이면 peft_config에서 어댑터 이름을 추출한다.
    """
    adapter_name = "default"
    # model이 PeftModel인 경우 어댑터 이름 추출
    if model is not None and hasattr(model, "peft_config"):
        names = list(getattr(model, "peft_config").keys())
        if names:
            adapter_name = names[0]

    fixed = {}
    for key, value in state_dict.items():
        if "lora_A.weight" in key:
            new_key = key.replace("lora_A.weight", f"lora_A.{adapter_name}.weight")
        elif "lora_B.weight" in key:
            new_key = key.replace("lora_B.weight", f"lora_B.{adapter_name}.weight")
        elif "lora_C.weight" in key:
            new_key = key.replace("lora_C.weight", f"lora_C.{adapter_name}.weight")
        else:
            new_key = key
        fixed[new_key] = value
    return fixed


from typing import Optional
import torch


def check_activation(model) -> bool:
    """LoRA 어댑터 활성 확인 - 안전 버전"""
    print("🔍 Checking adapter activation...")
    
    # PeftModel 확인
    if "PeftModel" not in type(model).__name__:
        print("   ❌ Not a PEFT model")
        return False
    
    # LoRA 파라미터 확인만으로 충분
    try:
        lora_count = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
        print(f"   📊 LoRA parameters found: {lora_count}")
        
        if lora_count > 0:
            print(f"   ✅ LoRA adapter is active")
            return True
        else:
            print(f"   ❌ No LoRA parameters")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False