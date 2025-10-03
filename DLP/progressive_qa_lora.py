"""
python progressive_qa_lora.py \
  --base_dir /home/devewha/ProgressivePruning_ver2/A \
  --bundles_dir /home/devewha/ProgressivePruning_ver2/bundles \
  --stage 1 \
  --out_adapters /home/devewha/ProgressivePruning_ver2/adapters \
  --qa_dataset squad --max_samples 16000 --seq_len 1024 --epochs 1 --bs 1 --grad_acc 16

python progressive_qa_lora.py \
  --base_dir /home/devewha/ProgressivePruning_ver2/A \
  --bundles_dir /home/devewha/ProgressivePruning_ver2/bundles \
  --stage 2 \
  --out_adapters /home/devewha/ProgressivePruning_ver2/adapters \
  --qa_dataset squad --max_samples 16000 --seq_len 1024 --epochs 1 --bs 1 --grad_acc 16

python progressive_qa_lora.py \
  --base_dir /home/devewha/ProgressivePruning_ver2/A \
  --bundles_dir /home/devewha/ProgressivePruning_ver2/bundles \
  --stage 3 \
  --out_adapters /home/devewha/ProgressivePruning_ver2/adapters \
  --qa_dataset squad --max_samples 16000 --seq_len 1024 --epochs 1 --bs 1 --grad_acc 16

# A 단계 (A LoRA: QA SFT)
python progressive_qa_lora.py \
  --base_dir /home/devewha/ProgressivePruning_ver2/A \
  --bundles_dir /home/devewha/ProgressivePruning_ver2/bundles \
  --stage 1 \
  --out_adapters /home/devewha/ProgressivePruning_ver2/adapters \
  --qa_dataset squad --max_samples 16000 --seq_len 1024 --epochs 3 --bs 1 --grad_acc 32

# B 단계 (A 고정 + B 복구 + B LoRA: QA SFT)
python progressive_qa_lora.py \
  --base_dir /home/devewha/ProgressivePruning_ver2/A \
  --bundles_dir /home/devewha/ProgressivePruning_ver2/bundles \
  --stage 2 \
  --out_adapters /home/devewha/ProgressivePruning_ver2/adapters \
  --qa_dataset squad --max_samples 16000 --seq_len 1024 --epochs 3 --bs 1 --grad_acc 32

# C 단계 (A,B 고정 + C 복구 + C LoRA: QA SFT)
python progressive_qa_lora.py \
  --base_dir /home/devewha/ProgressivePruning_ver2/A \
  --bundles_dir /home/devewha/ProgressivePruning_ver2/bundles \
  --stage 3 \
  --out_adapters /home/devewha/ProgressivePruning_ver2/adapters \
  --qa_dataset squad_v2 --max_samples 16000 --seq_len 1024 --epochs 3 --bs 1 --grad_acc 32

"""

#!/usr/bin/env python3
# progressive_qa_lora.py (fixed)

import os, json, torch
import re
import torch.nn.functional as F
from copy import deepcopy
from typing import List
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    default_data_collator, Trainer
)
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, PeftModel
from datetime import datetime, UTC
from peft.utils import get_peft_model_state_dict


def export_adapter_pt_and_recipe(model, out_dir, adapter_name, *, base_dir, bundles_dir, stage, trained_indices, tokenizer_dir=None):
    """
    - 어댑터 가중치만 .pt 단일 파일로 저장
    - 로딩 레시피(JSON) 함께 저장: 팀원이 쉽게 복원하도록
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) 어댑터 가중치 추출(.pt)
    state = get_peft_model_state_dict(model, adapter_name=adapter_name)

    #특정 레이어에 대한 어댑터 추출
    _LAYER_RE = re.compile(r"\blayers\.(\d+)\.")  # ...layers.21. ... 같은 키 포착

    def _filter_state_dict_by_layers(state_dict, keep_layers: set[int]):
        out = {}
        for k, v in state_dict.items():
            m = _LAYER_RE.search(k)
            if m and int(m.group(1)) in keep_layers:
                out[k] = v
        return out

    keep = set(int(i) for i in trained_indices)  # stage별 A_idx/B_idx/C_idx를 넘겨줌
    slim_state = _filter_state_dict_by_layers(state, keep)
    if not slim_state:
        print(f"[warn] slim_state is empty for adapter={adapter_name}. "
              f"Check regex or trained_indices. Falling back to full state.")
        slim_state = state

    pt_path = os.path.join(out_dir, f"{adapter_name}.pt")
    torch.save(slim_state, pt_path)

    # 2) 어댑터 설정/메타(JSON)
    raw_cfg = None
    try:
        if hasattr(model, "peft_config"):
            pc = model.peft_config  # peft_config may be PeftConfig or dict — try to extract adapter config if keyed
            if isinstance(pc, dict):
                raw_cfg = pc.get(adapter_name, None) or pc.get("default", None) or pc
            else:
                raw_cfg = pc
    except Exception:
        raw_cfg = None

    cfg = _peft_cfg_to_dict(raw_cfg) if raw_cfg is not None else None

    recipe = {
        "saved_at": datetime.now(UTC).isoformat(),
        "stage": stage,  # 1|2|3 or "A"|"B"|"C"
        "adapter_name": adapter_name,
        "adapter_pt": os.path.abspath(pt_path),
        "base_dir": os.path.abspath(base_dir),
        "bundles_dir": os.path.abspath(bundles_dir) if bundles_dir else None,
        "tokenizer_dir": os.path.abspath(tokenizer_dir) if tokenizer_dir else base_dir,
        "trained_layer_indices": sorted(list(map(int, trained_indices))),  # 이 스테이지에서 LoRA 학습된 레이어
        "peft_config": cfg,  # r, alpha, dropout, target_modules 등
        "load_instructions": [
            "1) base_dir에서 모델 로드 (AutoModelForCausalLM.from_pretrained)",
            "2) 같은 peft_config로 어댑터 생성(get_peft_model 또는 add_adapter)",
            "3) torch.load(adapter_pt)로 state_dict 로드(strict=False 권장)",
            "4) model.set_adapter(adapter_name) 로 활성화",
            "5) (B/C 필요 시) bundles_dir에서 해당 레이어 rehydrate 후 사용"
        ],
    }
    with open(os.path.join(out_dir, f"{adapter_name}_recipe.json"), "w", encoding="utf-8") as f:
        json.dump(recipe, f, ensure_ascii=False, indent=2)

    print(f"[export] adapter={adapter_name} → {pt_path}")
    print(f"[export] recipe → {os.path.join(out_dir, f'{adapter_name}_recipe.json')}")


def _peft_cfg_to_dict(cfg):
    # 이미 dict면 그대로
    if isinstance(cfg, dict):
        return cfg
    # peft >= 0.10 LoraConfig/PeftConfig는 to_dict 지원
    to_dict = getattr(cfg, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    # fallback: 객체 __dict__에서 json 가능한 것만 필터
    out = {}
    for k, v in getattr(cfg, "__dict__", {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            if all(isinstance(x, (str, int, float, bool)) or x is None for x in v):
                out[k] = list(v)
    return out


# ----------------------------
# Helpers (layers / devices)
# ----------------------------

# 🔧 ADD: robust layer container finder
def _get_layer_container(model):
    """
    Return the ModuleList of decoder layers for LLaMA/OPT across PEFT/Accelerate wrappers.
    """
    candidates = [
        "model.layers",
        "model.decoder.layers",
        "model.model.layers",
        "model.model.decoder.layers",
        "base_model.model.layers",
        "base_model.model.decoder.layers",
        "base_model.model.model.layers",
        "base_model.model.model.decoder.layers",
    ]
    for path in candidates:
        cur = model
        try:
            for seg in path.split("."):
                cur = getattr(cur, seg)
            # a ModuleList-like container?
            if hasattr(cur, "__len__") and hasattr(cur, "__getitem__"):
                return cur
        except Exception:
            continue
    raise AttributeError("Could not locate decoder layer container on model (checked: {})."
                         .format(", ".join(candidates)))


def _get_layers(model, is_opt: bool):
    return model.model.decoder.layers if is_opt else model.model.layers


def _assert_bundle_files_exist(bundles_dir: str, group: str, indices: list):
    """
    bundles_dir/group 에 layer_{idx:03d}.safetensors 파일이 모두 존재하는지 확인.
    존재하지 않으면 예외 발생(즉시 중단) — 안전장치.
    """
    group_dir = os.path.join(bundles_dir, group)
    if not os.path.isdir(group_dir):
        raise FileNotFoundError(f"[bundles] group dir not found: {group_dir}")

    missing = []
    zero_size = []
    for i in indices:
        fname = os.path.join(group_dir, f"layer_{int(i):03d}.safetensors")
        if not os.path.isfile(fname):
            missing.append(i)
        else:
            try:
                if os.path.getsize(fname) == 0:
                    zero_size.append(i)
            except OSError:
                zero_size.append(i)

    if missing or zero_size:
        msg = []
        if missing:
            msg.append(f"missing files for layers: {missing}")
        if zero_size:
            msg.append(f"zero-size files for layers: {zero_size}")
        raise FileNotFoundError(f"[bundles] problems in {group_dir}: " + "; ".join(msg))

    print(f"[bundles-ok] all {len(indices)} files present in {group_dir}")


def _find_embed_device(model):
    paths = ["model.embed_tokens","model.model.embed_tokens","base_model.model.model.embed_tokens"]
    for p in paths:
        cur = model
        try:
            for seg in p.split("."):
                cur = getattr(cur, seg)
            return next(cur.parameters()).device
        except Exception:
            pass
    return next(model.parameters()).device


"""
def _load_prev_adapters(model, adapters_root: str, names: List[str]):
    name2dir = {"stageA": "A_lora", "stageB": "B_lora", "stageC": "C_lora"}
    for i, nm in enumerate(names):
        adir = os.path.join(adapters_root, name2dir.get(nm, nm))
        if not os.path.isdir(adir):
            print(f"[adapter-load] missing {nm} at {adir} -> skip"); continue
        if i == 0 and not isinstance(model, PeftModel):
            model = PeftModel.from_pretrained(model, adir, local_files_only=True)
        else:
            model.load_adapter(adir, adapter_name=nm, local_files_only=True)
        print(f"[adapter-load] loaded {nm} from {adir}")
    for n,p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = False
    return model
"""


def _load_prev_adapters(model, adapters_root: str, names: List[str]):
    from peft import PeftModel
    name2dir = {
        "stageA": "A_lora",
        "stageB": "B_lora",
        "stageC": "C_lora",
        "A": "A_lora",
        "B": "B_lora",
        "C": "C_lora",
    }
    print("[debug] __file__ =", __file__)
    print("[debug] adapters_root =", os.path.abspath(adapters_root))
    for i, nm in enumerate(names):
        top = os.path.join(adapters_root, name2dir.get(nm, nm))  # .../adapters/A_lora
        inner = os.path.join(top, nm)                           # .../adapters/A_lora/stageA
        flat = os.path.join(adapters_root, nm)                  # .../adapters/stageA (레거시)
        candidates = [inner, top, flat]
        chosen = None
        for adir in candidates:
            print(f"[debug] try {adir} exists={os.path.isdir(adir)}")
            if os.path.isdir(adir):
                chosen = adir
                break
        if not chosen:
            raise FileNotFoundError(f"[adapter-load] cannot find {nm} under {adapters_root} "
                                    f"(tried: {candidates})")
        if i == 0 and not isinstance(model, PeftModel):
            model = PeftModel.from_pretrained(model, chosen, adapter_name=nm, local_files_only=True)
        else:
            model.load_adapter(chosen, adapter_name=nm, local_files_only=True)
        print(f"[adapter-load] loaded {nm} from {chosen}")

    # freeze all LoRA params loaded
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = False
    return model


# ----------------------------
# Keep-32 trick (PassLayer reinsert)
# ----------------------------
def _reapply_passlayers_from_manifest(model, base_dir: str):
    import json, os, torch.nn as nn
    man_path = os.path.join(base_dir, "manifest.json")
    print("[reapply] looking for:", man_path)
    if not os.path.isfile(man_path):
        print("[reapply] manifest.json not found -> skip")
        return model
    try:
        man = json.load(open(man_path, "r"))
    except Exception as e:
        print(f"[reapply] failed to read manifest: {e} -> skip")
        return model

    # 1) 다양한 스키마 지원: simdrop, top-level, stages.*
    removed = None
    removed = (man.get("simdrop", {}) or {}).get("removed_layers")
    if not removed:
        removed = man.get("removed_layers")
    if not removed:
        stages = man.get("stages", {})
        A_drop = (stages.get("A", {}) or {}).get("dropped_layers", [])
        B_rem = (stages.get("B", {}) or {}).get("removed_layers", [])
        C_rem = (stages.get("C", {}) or {}).get("removed_layers", [])
        # A 단계에서 빈자리를 메워야 하므로, A에서 드랍된(= B,C 통합) 전 레이어를 대상으로 패스레이어 적용
        removed = A_drop or sorted(set(B_rem + C_rem))

    if not removed:
        print("[reapply] removed_layers empty (checked simdrop/top-level/stages) -> skip")
        return model

    # ---- PassLayer 선택 (프로젝트 커스텀 있으면 사용, 없으면 SafePass) ----
    try:
        from lib.identity import LlamaPassLayer as _Inner
        class _Wrapper(nn.Module):
            def __init__(self, hidden):
                super().__init__(); self.inner = _Inner(hidden)
            def forward(self, hidden_states, *a, **kw):
                out = self.inner(hidden_states, *a, **kw)
                return out[0] if isinstance(out, tuple) else out
        def _make(h): return _Wrapper(h)
        print("[reapply] using project LlamaPassLayer")
    except Exception:
        class SafePass(nn.Module):
            def __init__(self, hidden):
                super().__init__()
            def forward(self, x, *a, **kw):
                return x
        def _make(h): return SafePass(h)
        print("[reapply] using SafePassLayer")

    """
    # ---- 레이어 컨테이너 찾기 (더 튼튼하게) ----
    def _get_layer_container(model):
        cands = [
            "model.layers",
            "model.decoder.layers",
            "model.model.layers",
            "model.model.decoder.layers",
            "base_model.model.layers",
            "base_model.model.decoder.layers",
            "base_model.model.model.layers",
            "base_model.model.model.decoder.layers",
        ]
        for path in cands:
            cur = model
            try:
                for seg in path.split("."):
                    cur = getattr(cur, seg)
                if hasattr(cur, "__len__") and hasattr(cur, "__getitem__"):
                    return cur
            except Exception:
                pass
        raise AttributeError("Could not locate decoder layer container.")
    """

    try:
        layers = _get_layer_container(model)
    except Exception as e:
        print("[reapply] cannot locate layers:", e, "-> skip")
        return model

    L = len(layers)
    hidden = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden is None:
        # LLaMA에서 통상적으로 접근 가능: 첫 레이어의 dim 추정
        try:
            hidden = model.config.hidden_size
        except Exception:
            print("[reapply] hidden_size not found -> skip")
            return model

    for i in removed:
        if 0 <= int(i) < L:
            layers[int(i)] = _make(hidden)
        else:
            print(f"[reapply] index {i} out of range (0..{L-1}) -> skip this one")

    print(f"[reapply] installed PassLayer on: {sorted(map(int, removed))}")
    return model


# ----------------------------
# Rehydrate dropped layers (from bundles)
# ----------------------------
# --- progressive_qa_lora.py: _rehydrate_layers() 패치 ---
# progressive_qa_lora.py : _rehydrate_layers()
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

def _rehydrate_layers(model, bundle_dir: str, indices: List[int]):
    layers = _get_layer_container(model)
    dtype = next(model.parameters()).dtype
    tgt = next(model.parameters()).device  # ← 단일 디바이스로 고정
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


# ----------------------------
# PEFT helpers
# ----------------------------
def _freeze_all(model):
    for _, p in model.named_parameters():
        p.requires_grad = False


def _attach_new_adapter(model, name: str,
                        target_modules=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"),
                        r=8, alpha=16, dropout=0.05):
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )
    if isinstance(model, PeftModel):
        if name in getattr(model, "peft_config", {}):
            model.set_adapter(name)
        else:
            model.add_adapter(name, cfg)
        model.set_adapter(name)
        return model
    return get_peft_model(model, cfg, adapter_name=name)


"""
# ❌ old:
def _enable_only_lora_on_indices_for_adapter(model, indices: List[int], adapter_name: str, is_opt: bool):
# ✅ new:
"""
def _enable_only_lora_on_indices_for_adapter(model, indices: List[int], adapter_name: str):
    # 1) all off
    for _, p in model.named_parameters():
        p.requires_grad = False

    layers = _get_layer_container(model)
    enabled_sites, enabled_params = 0, 0
    for li in indices:
        if li < 0 or li >= len(layers):
            continue
        for _, m in layers[li].named_modules():
            if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
                if adapter_name in getattr(m, "lora_A", {}):
                    for p in m.lora_A[adapter_name].parameters():
                        p.requires_grad = True
                        enabled_params += p.numel()
                        enabled_sites += 1
                if adapter_name in getattr(m, "lora_B", {}):
                    for p in m.lora_B[adapter_name].parameters():
                        p.requires_grad = True
                        enabled_params += p.numel()
                        enabled_sites += 1

    print(f"[trainable] adapter={adapter_name} layers={indices} -> {enabled_sites} sites, {enabled_params} params")
    if enabled_sites == 0:
        # helpful debug: show available adapter keys in one layer
        sample_keys = []
        for m in layers[indices[0]].modules():
            if hasattr(m, "lora_A"):
                sample_keys = list(getattr(m, "lora_A").keys())
            if sample_keys:
                break
        raise RuntimeError(
            f"No LoRA params enabled for adapter='{adapter_name}'. "
            f"Available adapter keys (layer {indices[0]}): {sample_keys}"
        )


def _enable_only_lora_on_indices_for_adapter_by_name(model, indices: List[int], adapter_name: str, keep_layernorm=False):
    # 1) 기본: 모든 파라미터 비활성화
    for n, p in model.named_parameters():
        p.requires_grad = False

    enabled = 0
    layer_patterns = [f"model.layers.{i}." for i in indices]
    for pname, p in model.named_parameters():
        # (a) 대상 레이어 + LoRA 파라미터만 활성
        if any(pat in pname for pat in layer_patterns) and ("lora_" in pname.lower() or "lora" in pname.lower()):
            if adapter_name is None or adapter_name.lower() in pname.lower():
                p.requires_grad = True
                enabled += p.numel()
            continue
        # (b) keep_layernorm=True면, 대상 레이어의 LN만 활성 (전역 LN 금지!)
        if keep_layernorm:
            if any(pat in pname for pat in layer_patterns) and ("layernorm" in pname.lower() or ".ln_" in pname.lower() or ".norm" in pname.lower()):
                p.requires_grad = True
                continue

    if enabled == 0:
        raise RuntimeError(f"No LoRA params enabled for adapter='{adapter_name}' on layers={indices}.")
    print(f"[trainable] adapter={adapter_name} layers={indices} -> enabled params count {enabled}")


# ----------------------------
# QA dataset (SQuAD / SQuAD v2) → SFT labels(mask prompt)
# ----------------------------
def _build_prompt(context: str, question: str, unans_token="unanswerable"):
    return (
        "You are a helpful QA assistant. Answer the question using the context. "
        f"If the answer is not in the context, say '{unans_token}'.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )


def _load_qa_sft_dataset(
    tokenizer,
    qa_dataset="squad",  # 'squad' | 'squad_v2'
    split="train",
    max_samples=5000,
    seq_len=1024,
    unans_token="unanswerable",
    add_eos=True
):
    ds = load_dataset(qa_dataset, split=split)
    if max_samples:
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def to_ex(ex):
        ctx = ex.get("context","")
        q = ex.get("question","")
        ans_list = ex.get("answers",{}).get("text",[])
        target = (ans_list[0] if ans_list else ("unanswerable" if qa_dataset=="squad_v2" else ""))

        prompt = _build_prompt(ctx, q, unans_token)
        full = prompt + " " + target + ("\n" if add_eos else "")

        enc = tokenizer(full, truncation=True, max_length=seq_len, padding="max_length")
        p_enc = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=seq_len, padding="max_length")

        input_ids = enc["input_ids"]
        attn = enc["attention_mask"]

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # 길이 계산(패딩 방향 무시하고 동작)
        full_nonpad = sum(attn)  # 전체 non-pad 토큰 수
        prompt_len = sum(1 for x in p_enc["input_ids"] if x != pad_id)

        # 답변 길이 부족하면 드롭 플래그
        drop = int((full_nonpad - prompt_len) < 1)

        # 라벨 초기화 + pad 위치 -100
        labels = enc["input_ids"][:]
        for i, a in enumerate(attn):
            if a == 0:
                labels[i] = -100

        # 첫 non-pad 위치(start)를 찾아 프롬프트만 -100
        # (left padding이면 start > 0, right padding이면 start == 0)
        try:
            start = next(i for i, a in enumerate(attn) if a == 1)
        except StopIteration:
            # 전부 pad면 그냥 드롭
            drop = 1
            start = 0
        end_prompt = min(start + prompt_len, len(labels))
        for i in range(start, end_prompt):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "__drop__": drop,
        }

    ds = ds.map(to_ex, remove_columns=ds.column_names)
    # drop Nones
    ds = ds.filter(lambda x: x["__drop__"] == 0)
    if "__drop__" in ds.column_names:
        ds = ds.remove_columns(["__drop__"])
    return ds


# ----------------------------
# Train
# ----------------------------
def train_lora(model, tokenizer, out_dir: str, train_ds, lr=2e-4, epochs=1, bs=4, grad_acc=8, fp16=True, adapter_name=None):
    os.makedirs(out_dir, exist_ok=True)
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=20,
        save_strategy="no",
        fp16=fp16,  # 안전 옵션
        warmup_ratio=0.1,  # 워밍업
        max_grad_norm=1.0,  # 클리핑: gradient_clip_val 대신 이걸 사용
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=default_data_collator  # labels already provided
    )
    trainer.train()

    if isinstance(model, PeftModel):
        try:
            model.save_pretrained(out_dir, selected_adapters=[adapter_name] if adapter_name else None)
        except TypeError:
            model.save_pretrained(out_dir)
    else:
        print("[warn] model is not PeftModel; adapter save may be skipped")


# ----------------------------
# Main
# ----------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, required=True, help="프루닝된 A 모델 디렉토리")
    ap.add_argument("--bundles_dir", type=str, required=True, help="bundles/B, bundles/C 루트")
    ap.add_argument("--stage", type=int, choices=[1,2,3], required=True, help="1:A-LoRA, 2:B-LoRA, 3:C-LoRA")
    ap.add_argument("--out_adapters", type=str, default="./adapters")

    # QA options
    ap.add_argument("--qa_dataset", type=str, choices=["squad","squad_v2"], default="squad")
    ap.add_argument("--max_samples", type=int, default=5000)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--unans_token", type=str, default="unanswerable")

    # Train hparams
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--grad_acc", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_dir, use_fast=True, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # 단일 GPU면 device_map=None 권장(Trainer와 안정성↑)
    device = torch.device(os.environ.get("DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_dir, torch_dtype=torch.float16, device_map=None, local_files_only=True
    )
    model.to(device)  #CPU로 이동
    model.config.use_cache = False

    # keep 32 logical layers for A stage convenience
    model = _reapply_passlayers_from_manifest(model, args.base_dir)

    is_opt = "opt" in model.config.model_type.lower()

    with open(os.path.join(args.base_dir, "prune_log.json"), "r", encoding="utf-8") as f:
        log = json.load(f)
    B_idx, C_idx = log["split"]["B"], log["split"]["C"]

    # QA SFT dataset (prompt→answer, prompt tokens masked)
    ds = _load_qa_sft_dataset(
        tok,
        qa_dataset=args.qa_dataset,
        split="train",
        max_samples=args.max_samples,
        seq_len=args.seq_len,
        unans_token=args.unans_token
    )

    if args.stage == 1:
        removed = set(B_idx) | set(C_idx)
        all_idx = list(range(getattr(model.config, "num_hidden_layers", len(_get_layers(model, is_opt)))))
        A_idx = [i for i in all_idx if i not in removed]

        model = _attach_new_adapter(model, "stageA")
        model.set_adapter("stageA")

        _freeze_all(model)
        #_enable_only_lora_on_indices_for_adapter(model, A_idx, "stageA")
        _enable_only_lora_on_indices_for_adapter_by_name(model, A_idx, "stageA", keep_layernorm=False)

        out_dir = os.path.join(args.out_adapters, "A_lora")
        train_lora(model, tok, out_dir, ds, args.lr, args.epochs, args.bs, args.grad_acc, adapter_name="stageA")

        export_adapter_pt_and_recipe(
            model, out_dir, "stageA",
            base_dir=args.base_dir, bundles_dir=args.bundles_dir, stage="A",
            trained_indices=A_idx, tokenizer_dir=args.base_dir
        )

    elif args.stage == 2:
        model = _load_prev_adapters(model, args.out_adapters, names=["stageA"])

        #--- 추가: B 번들 파일 존재 검사 (rehydrate 전에) ---
        _assert_bundle_files_exist(args.bundles_dir, "B", B_idx)

        # B 레이어 복원
        _rehydrate_layers(model, os.path.join(args.bundles_dir, "B"), B_idx)

        # (2) 여기서 '실레이어' 여부 검사 (PassLayer가 아닌지)
        layers = _get_layer_container(model)
        bad = [i for i in B_idx if not isinstance(layers[i], LlamaDecoderLayer)]  # B는 실레이어, C는 pass인지 빠르게 점검
        badB = [i for i in B_idx if not isinstance(layers[i], LlamaDecoderLayer)]
        if badB:
            raise RuntimeError(f"[check] B indices not real LlamaDecoderLayer: {badB}")

        def _is_pass(m):
            return m.__class__.__name__.lower().find("llamadecoderlayer") == -1
        badC = [i for i in C_idx if not _is_pass(layers[i])]
        if badC:
            print(f"[warn] C 중 실레이어가 섞여 있음(패스 예상): {badC}")

        # 어댑터 부착 및 학습
        model = _attach_new_adapter(
            model, "stageB",
            target_modules=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"),
            r=8, alpha=16, dropout=0.05
        )
        model.set_adapter("stageB")

        # B 한 레이어 내부에서 lora가 실제 붙었는지
        L = _get_layer_container(model)[B_idx[0]]
        keys = [n for n,_ in L.named_parameters() if "lora_A" in n or "lora_B" in n]
        print("[probe] sample lora keys in layer", B_idx[0], ":", keys[:10])

        # 전체 동결 → B 레이어의 LoRA만 활성
        _freeze_all(model)
        # 우선 안전성을 위해 keep_layernorm = False로 두기
        _enable_only_lora_on_indices_for_adapter_by_name(model, B_idx, "stageB", keep_layernorm=False)

        # (a) non-LoRA 강제 동결(혹시라도 풀린 파라미터 잠그기)
        for n, p in model.named_parameters():
            if "lora_" not in n.lower() and "lora" not in n.lower():
                p.requires_grad = False

        # (b) trainable 요약
        trainable = [(n,p.numel()) for n,p in model.named_parameters() if p.requires_grad]
        print("[post-fix] total trainable params:", sum(x[1] for x in trainable))

        # (c) LoRA 파라미터 유한성 검사
        bad_keys = []
        for n, p in model.named_parameters():
            if ("lora_" in n.lower() or "lora" in n.lower()) and p.numel() > 0:
                if not torch.isfinite(p).all():
                    bad_keys.append(n)
        if bad_keys:
            print("[FATAL] NaN/Inf in LoRA params:", bad_keys[:20])
            raise RuntimeError("Detected NaN/Inf in LoRA parameters — aborting.")
        print("[OK] LoRA finite & non-LoRA frozen for stageB")

        out_dir = os.path.join(args.out_adapters, "B_lora")
        train_lora(model, tok, out_dir, ds, lr=args.lr, epochs=args.epochs, bs=args.bs, grad_acc=args.grad_acc, fp16=True, adapter_name="stageB")

        export_adapter_pt_and_recipe(
            model, out_dir, "stageB",
            base_dir=args.base_dir, bundles_dir=args.bundles_dir, stage="B",
            trained_indices=B_idx, tokenizer_dir=args.base_dir
        )

    elif args.stage == 3:
        # A 어댑터 로드, merge
        model = _load_prev_adapters(model, args.out_adapters, names=["stageA","stageB"])
        model.set_adapter("stageA")
        if hasattr(model, "merge_and_unload"):
            print("[freeze-A] merging stageA into base...")
            model = model.merge_and_unload()  # A 효과를 base에 고정
            model.config.use_cache = False
            model.to(next(model.parameters()).device)
        else:
            print("[warn] merge_and_unload unavailable; A를 고정하지 못했을 수 있음(권장: peft>=0.10)")

        # 1) B, C 레이어 모두 복원(실레이어 장착)
        _assert_bundle_files_exist(args.bundles_dir, "B", B_idx)
        _assert_bundle_files_exist(args.bundles_dir, "C", C_idx)
        _rehydrate_layers(model, os.path.join(args.bundles_dir, "B"), B_idx)
        _rehydrate_layers(model, os.path.join(args.bundles_dir, "C"), C_idx)

        # (2) 복원 검증 (B, C 각각)
        layers = _get_layer_container(model)
        badB = [i for i in B_idx if layers[i].__class__.__name__.lower().find("llamadecoderlayer") == -1]
        badC = [i for i in C_idx if layers[i].__class__.__name__.lower().find("llamadecoderlayer") == -1]
        print("[check] non-real layers in B:", badB, " / C:", badC)
        if badB or badC:
            raise RuntimeError(f"Non-real layers detected. B:{badB}, C:{badC}. Check bundles and indices.")

        # (3) B 어댑터 로드->병합 및 고정
        model = _load_prev_adapters(model, args.out_adapters, names=["stageB"])
        model.set_adapter("stageB")
        if hasattr(model, "merge_and_unload"):
            print("[freeze-B] merging stageB into base...")
            model = model.merge_and_unload()  # B 효과를 base에 고정
            model.config.use_cache = False
            model.to(next(model.parameters()).device)
        else:
            print("[warn] merge_and_unload unavailable; B를 고정하지 못했을 수 있음")

        # (4) C 어댑터 장착 및 학습
        model = _attach_new_adapter(
            model, "stageC",
            target_modules=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"),
            r=8, alpha=16, dropout=0.05
        )
        model.set_adapter("stageC")
        _freeze_all(model)
        _enable_only_lora_on_indices_for_adapter_by_name(model, C_idx, "stageC", keep_layernorm=True)

        # (안전) 비-LoRA는 다시 잠그기 + 유한성 검사
        for n, p in model.named_parameters():
            if "lora" not in n.lower():
                p.requires_grad = False
        for n, p in model.named_parameters():
            if "lora" in n.lower() and p.numel() > 0 and not torch.isfinite(p).all():
                raise RuntimeError(f"[FATAL] NaN/Inf detected pre-train in {n}")

        out_dir = os.path.join(args.out_adapters, "C_lora")
        train_lora(model, tok, out_dir, ds, args.lr, args.epochs, args.bs, args.grad_acc, fp16=True, adapter_name="stageC")

        export_adapter_pt_and_recipe(
            model, out_dir, "stageC",
            base_dir=args.base_dir, bundles_dir=args.bundles_dir, stage="C",
            trained_indices=C_idx, tokenizer_dir=args.base_dir
        )

    else:
        raise ValueError("Invalid stage")


if __name__ == "__main__":
    main()
