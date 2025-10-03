# check_model_size.py
import os
import math
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

def human(n):
    # bytes -> human readable (binary GiB)
    for unit in ["B","KiB","MiB","GiB","TiB"]:
        if abs(n) < 1024.0:
            return f"{n:3.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PiB"

def folder_bytes(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total

def sum_checkpoint_files(path, exts=(".bin", ".safetensors", ".pt", ".ckpt")):
    total = 0
    files = []
    for fname in os.listdir(path):
        if fname.endswith(exts):
            p = os.path.join(path, fname)
            s = os.path.getsize(p)
            total += s
            files.append((fname, s))
    return total, files

def analyze_checkpoint(path):
    # 1) 디스크 크기(폴더 전체)와 체크포인트 파일 크기 합
    print("== Disk sizes ==")
    if os.path.isdir(path):
        total_bytes = folder_bytes(path)
        print("Folder total:", human(total_bytes), f"({total_bytes:,} bytes)")
        csum, files = sum_checkpoint_files(path)
        if csum > 0:
            print("Checkpoint files total:", human(csum), f"({csum:,} bytes)")
            for fname, s in sorted(files, key=lambda x:-x[1])[:10]:
                print("  ", fname, human(s))
        else:
            print("No typical checkpoint files (.bin/.safetensors) found in dir, showing top files:")
            all_files = sorted([(f, os.path.getsize(os.path.join(path,f))) for f in os.listdir(path)], key=lambda x:-x[1])[:10]
            for f,s in all_files:
                print("  ", f, human(s))
    else:
        # path may be a single file
        if os.path.isfile(path):
            s = os.path.getsize(path)
            print("File:", path, human(s), f"({s:,} bytes)")
        else:
            print("Path not found:", path)
            return

    # 2) (선택) 모델 로드해서 파라미터 개수와 nonzero 체크
    print("\n== Parameter stats (requires loading model into CPU) ==")
    try:
        print("Loading model (device_map='cpu', low_cpu_mem_usage=True)...")
        model = AutoModelForCausalLM.from_pretrained(path, device_map="cpu", low_cpu_mem_usage=True)
    except Exception as e:
        print("Model load failed:", e)
        print("Skipping detailed param analysis.")
        return

    total = 0
    nonzero = 0
    dtype_counts = {}
    for n, p in model.named_parameters():
        num = p.numel()
        total += num
        # move to CPU (should already be), ensure no grad, use .data
        arr = p.data
        nz = int((arr != 0).sum().item())
        nonzero += nz
        dt = str(arr.dtype)
        dtype_counts[dt] = dtype_counts.get(dt, 0) + num

    print("Total params:", f"{total:,}")
    print("Non-zero params:", f"{nonzero:,}", f"({100.0 * nonzero / total:3.3f}%)")
    # dtype summary
    print("Dtype counts:")
    for k,v in dtype_counts.items():
        print(" ", k, f": {v:,} params")

    # theoretical memory footprints
    print("\n== Theoretical memory sizes (parameter bytes only, not optimizer etc.)")
    # pick 32/16/2-byte assumptions
    for bps, name in [(4,"float32 (4B)"), (2,"float16 (2B)"), (1,"int8/4-bit approx per param?")]:
        byte_size = total * bps
        print(f" {name:15s} : {human(byte_size)} ({byte_size:,} bytes)")

    # effective saved size if using nonzero count (i.e., storage of dense arrays)
    # note: sparse storage formats change this calculation — this is just param-count-based estimate
    for bps, name in [(4,"float32"), (2,"float16")]:
        full = total * bps
        eff = nonzero * bps
        print(f" Effective if storing only nonzero ({name}): {human(eff)}  ({100.0*nonzero/total:3.3f}% of dense)")

    # cleanup
    del model
    import gc; gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="model folder or file path (HF folder)")
    args = parser.parse_args()
    analyze_checkpoint(args.path)
