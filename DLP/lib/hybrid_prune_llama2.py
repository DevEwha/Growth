# DLP + Angular-distance 결합 하이브리드 프루닝 엔트리

import argparse
import os
from types import SimpleNamespace
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
#from accelerate import dispatch_model
#from accelerate.utils import infer_auto_device_map, get_balanced_memory

from .data import get_loaders
from .prune import get_dlp_ratios, prune_sparsegpt_dlp, prune_wanda_dlp, MaskRecorder
from .simdrop import choose_block_to_drop, drop_consecutive_layers
from .healing import try_lora_heal
from .bundler import export_layer_bundle, split_indices



def _resolve_embed_device(model, fallback_device: str):
    """
    DLP 코드가 model.hf_device_map을 사용하므로, 여기서 임베딩 디바이스를 알아낸다.
    device_map='auto'로 로드했다면 'model.embed_tokens'가 있을 것.
    """
    embed_key = "model.embed_tokens"
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict) and embed_key in model.hf_device_map:
        dev = model.hf_device_map[embed_key]
        # dev가 'cuda:0' 같은 str일 수도, torch.device일 수도 있음
        return torch.device(dev) if not isinstance(dev, torch.device) else dev
    return torch.device(fallback_device)


def _load_model(model_name: str, seqlen: int, use_auto_map: bool, device_str: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    """ if use_auto_map:
        # Accelerate 기반 배치 (권장) → hf_device_map 생성됨
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    else:
        # 전체 단일 디바이스 (hf_device_map이 없을 수 있음 → DLP 코드 일부에서 가드 필요)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=None,
        ).to(device_str) """
    # 단일 GPU 강제
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,
        attn_implementation="eager",  # ← 추가
    ).to(device_str)

    model.seqlen = seqlen
    model.config.use_cache = False
    model.eval()
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--drop_frac", type=float, default=0.35)
    ap.add_argument("--keep_last_layer", action="store_true")
    ap.add_argument("--nsamples", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--sparsity_ratio", type=float, default=0.70)
    ap.add_argument("--alpha", type=float, default=0.12)
    ap.add_argument("--backend", type=str, choices=["sparsegpt", "wanda"], default="sparsegpt")
    ap.add_argument("--heal", action="store_true")
    ap.add_argument("--max_batches", type=int, default=64)
    ap.add_argument("--save_masks", type=str, default=None, help="프루닝 마스크 저장 경로(.pt). 비트팩은 .bitpt 권장")
    ap.add_argument("--save_dir", type=str, default="./hybrid_llama2_7b_pruned")
    # 로딩 전략
    ap.add_argument("--use_device_map_auto", action="store_true", help="권장: Accelerate로 device_map='auto' 사용")
    ap.add_argument("--save_removed_dir", type=str, default="./bundles", help="제거 레이어 번들(B/C) 저장 루트")
    ap.add_argument("--split_policy", type=str, choices=["half", "ratio"], default="half")
    ap.add_argument("--split_ratio", type=float, default=0.5, help="--split_policy=ratio 일 때만 사용")
    ap.add_argument("--prune_log", type=str, default="prune_log.json", help="어떤 레이어가 삭제(치환)됐는지 기록")

    args = ap.parse_args()
    torch.manual_seed(args.seed)

    # 1) 로드 (권장: --use_device_map_auto 로 hf_device_map 생성)
    model, tokenizer = _load_model(args.model, args.seqlen, args.use_device_map_auto, args.device)
    is_opt = "opt" in args.model.lower()
    embed_dev = _resolve_embed_device(model, args.device)

    # 2) 캘리브레이션 로더 (C4)
    print("[2/5] Calibration loader (C4)")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=args.seqlen, tokenizer=tokenizer)

    # 3) 연속 블록 드랍 찾기 → 드랍
    print("[3/5] Angular-distance block selection")

    L = len(model.model.decoder.layers) if is_opt else len(model.model.layers)
    n = int(round(args.drop_frac * L))
    
    if n > 0:
        # 3-1) 최적 연속 블록 선택
        best_ell, best_d, L_old = choose_block_to_drop(
            model, dataloader, embed_dev, n=n, is_opt=is_opt,
            keep_last_layer=args.keep_last_layer, max_batches=args.max_batches
        )
        print(f"→ L={L_old}, n={n}, ell*={best_ell}, d={best_d:.4f}")

        # 3-2) 제거 인덱스 & B/C 분할
        removed_indices = list(range(best_ell, best_ell + n))
        B_idx, C_idx = split_indices(
            removed_indices, policy=args.split_policy, ratio=args.split_ratio
        )

        # 3-3) 원본 레이어 번들(B/C) 저장 — 반드시 치환 전에!
        os.makedirs(args.save_removed_dir, exist_ok=True)
        export_layer_bundle(model, B_idx, os.path.join(args.save_removed_dir, "B"), is_opt, model.config)
        export_layer_bundle(model, C_idx, os.path.join(args.save_removed_dir, "C"), is_opt, model.config)
        print(f"→ Saved removed bundles: B({len(B_idx)}), C({len(C_idx)}) in {args.save_removed_dir}")

        # 3-4) 모델에 드랍 적용 (치환/삭제) — 한 번만!
        model = drop_consecutive_layers(model, best_ell, n, is_opt=is_opt)
        model = model.to(embed_dev)
        torch.cuda.empty_cache()

        new_depth = len(model.model.decoder.layers) if is_opt else len(model.model.layers)
        print(f"→ New depth (logical): {new_depth}")

        # 3-5) 로그 기록
        os.makedirs(args.save_dir, exist_ok=True)  # <— 누락되어 있던 부분
        prune_log = {
            "model": args.model,
            "seed": args.seed,
            "seqlen": args.seqlen,
            "drop_frac": args.drop_frac,
            "keep_last_layer": args.keep_last_layer,
            "selected_block": {
                "start": best_ell, "n": n, "indices": removed_indices,
                "angular_distance": float(best_d),
            },
            "split": {"policy": args.split_policy, "ratio": args.split_ratio, "B": B_idx, "C": C_idx},
        }
        with open(os.path.join(args.save_dir, args.prune_log), "w", encoding="utf-8") as f:
            json.dump(prune_log, f, ensure_ascii=False, indent=2)
        print(f"→ Logged: {os.path.join(args.save_dir, args.prune_log)}")
    else:
        print("→ Skipping layer drop (drop_frac=0)")
        
