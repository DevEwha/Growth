# apply_heal.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.healing import try_lora_heal  # 같은 디렉토리에 healing.py가 있어야 함
import os

PRUNED_DIR = "./hybrid_sparse35"        # 변경: 네 프루닝된 모델 디렉토리
DEVICE = "cuda:0"                # GPU
HEAL_TEXTS = [
    "Short calibration sentence to adapt the splice after layer dropping.",
    "Another small healing batch for continuity.",
    "A third brief example to stabilize transitions after layer removal."
] * 5  # 반복해서 충분한 샘플 수 확보 (총 15개)

# Hyperparams
r = 16
alpha = 16
lr = 1e-4
steps = 200   # 50~500 사이에서 조절. 메모리/시간 고려.

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device(DEVICE)

    print("Loading pruned model and tokenizer from:", PRUNED_DIR)
    model = AutoModelForCausalLM.from_pretrained(PRUNED_DIR, torch_dtype=torch.float16, device_map=None)
    tokenizer = AutoTokenizer.from_pretrained(PRUNED_DIR, use_fast=False)

    # model이 큰 경우 single-GPU로 올려야 함
    model = model.to(device)
    model.eval()

    # set seqlen if needed
    model.seqlen = getattr(model, "seqlen", 1024)

    print("Applying LoRA heal (this will wrap model with PEFT) ...")
    model = try_lora_heal(model, tokenizer, HEAL_TEXTS, device=device, r=r, alpha=alpha, lr=lr, steps=steps)

    # 옵션: PEFT 어댑터 병합(merge) — 가능하면 병합해 최종 dense 모델로 만들기
    try:
        if hasattr(model, "merge_and_unload"):
            print("Merging LoRA adapters into base weights (merge_and_unload)...")
            model = model.merge_and_unload()
        else:
            print("No merge_and_unload() available in this PEFT version — will save adapter+base separately.")
    except Exception as e:
        print("Merge failed or not supported:", e)

    SAVE_DIR = PRUNED_DIR + "_healed"
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("Saving healed model to", SAVE_DIR)
    model.save_pretrained(SAVE_DIR, safe_serialization=True)
    tokenizer.save_pretrained(SAVE_DIR)
    print("Done. Healed model saved.")

if __name__ == "__main__":
    main()
