import time
import torch
import torch.nn as nn
import numpy as np

from .sparsegpt import SparseGPT
from .layerwrapper import WrappedGPT
from .data import get_loaders
from .utils import find_layers, prepare_calibration_input_opt, return_given_alpha, prepare_calibration_input, check_outlier_mean
from .quant import *

from pdb import set_trace as st

# 파일 상단(임포트 아래)에 추가)
class MaskRecorder:
    """
    레이어별/서브모듈별 프루닝 위치(마스크)를 기록.
    - masks[key] = bool Tensor (True = 이번 프루닝으로 0이 된 위치)
    - key: f"{layer_idx}:{subname}" (예: "3:q_proj")
    """
    def __init__(self):
        self.masks = {}  # Dict[str, BoolTensor]
        self.meta = {}   # 원래 shape 저장 등 (압축/복구용)

    def add(self, layer_idx: int, subname: str, mask: torch.Tensor, weight_shape=None):
        key = f"{layer_idx}:{subname}"
        m = mask.detach().to("cpu", non_blocking=True).bool()
        self.masks[key] = m
        if weight_shape is not None:
            self.meta[key] = {"shape": tuple(weight_shape)}

    def add_from_weights(self, layer_idx: int, subname: str, before: torch.Tensor, after: torch.Tensor):
        # "이번 프루닝으로 새로 0이 된" 위치만 기록
        pre_nz = before.detach().ne(0).to("cpu")
        post_z = after.detach().eq(0).to("cpu")
        new_pruned = pre_nz & post_z
        self.add(layer_idx, subname, new_pruned, weight_shape=after.shape)

    def save(self, path: str):
        torch.save({"masks": self.masks, "meta": self.meta}, path)

    # 선택: 비트팩 저장(마스크 압축)
    def save_bitpacked(self, path: str):
        packed = {k: torch.packbits(v.flatten().to(torch.uint8)) for k, v in self.masks.items()}
        shapes = {k: v.shape for k, v in self.masks.items()}
        torch.save({"packed": packed, "shapes": shapes}, path)

    @staticmethod
    def load_bitpacked(path: str):
        obj = torch.load(path, map_location="cpu")
        packed, shapes = obj["packed"], obj["shapes"]
        masks = {}
        for k, p in packed.items():
            numel = int(torch.tensor(shapes[k]).prod().item())
            u = torch.unpackbits(p)[:numel].bool()
            masks[k] = u.view(*shapes[k])
        return {"masks": masks, "shapes": shapes}


def _maybe_to(x, dev):
    return x.to(dev) if (x is not None and hasattr(x, "to")) else x

# === RoPE-safe forward helpers (drop-in) ===
def _get_rope_cos_sin(layer, parent_model, x, position_ids):
    """
    HF 버전별로 RoPE 모듈 위치/시그니처가 다르므로 폭넓게 탐색해서 cos/sin 생성.
    시도 순서:
      1) layer.self_attn.rotary_emb / rope
      2) parent_model.model.rotary_emb / rope
      3) parent_model.rotary_emb / rope
    """
    candidates = []
    attn = getattr(layer, "self_attn", None)
    if attn is not None:
        for name in ("rotary_emb", "rope"):
            if hasattr(attn, name):
                candidates.append(getattr(attn, name))

    m = getattr(parent_model, "model", parent_model)
    for name in ("rotary_emb", "rope"):
        if hasattr(m, name):
            candidates.append(getattr(m, name))
    for name in ("rotary_emb", "rope"):
        if hasattr(parent_model, name):
            candidates.append(getattr(parent_model, name))

    # 시그니처가 제각각이라 여러 형태로 호출해본다.
    for rope in candidates:
        try:
            return rope(x, seq_len=x.shape[1])
        except TypeError:
            pass
        try:
            return rope(x, position_ids=position_ids)
        except Exception:
            pass
    return None, None


def _layer_forward_llama_safe(layer, parent_model, x, attention_mask, position_ids):
    """
    Llama 레이어 호환 forward:
    - 구버전: position_ids 로 호출
    - 신버전: position_embeddings=(cos, sin) 필요 -> 상위 모델에서 RoPE 찾아서 생성
    - 마지막 폴백: 일부 포크는 내부에서 자체 처리
    """
    # 1) 구버전 경로
    try:
        return layer(x, attention_mask=attention_mask, position_ids=position_ids)[0]
    except Exception:
        pass

    # 2) 신버전: cos/sin 만들어 넘김
    cos, sin = _get_rope_cos_sin(layer, parent_model, x, position_ids)
    if (cos is not None) and (sin is not None):
        # cos/sin을 입력과 같은 디바이스로 강제 이동
       cos = cos.to(x.device)
       sin = sin.to(x.device)
       return layer(x, attention_mask=attention_mask, position_embeddings=(cos, sin))[0]

    # 3) 최후 폴백
    return layer(x, attention_mask=attention_mask)[0]
# === end helpers ===

def _safe_sqrt(x):
    # 음수/NaN/Inf 방지
    return torch.sqrt(torch.clamp(x, min=0)).float()

def _safe_w_metric(weight, scaler_row):
    """
    |W| * sqrt(scaler_row)를 안전하게 계산.
    scaler_row의 NaN/Inf/음수/shape 문제도 여기서 흡수.
    """
    W = torch.abs(weight).float()
    if scaler_row is None:
        return torch.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)

    s = scaler_row.reshape(1, -1).to(W.device)
    s = _safe_sqrt(s)
    s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    M = W * s
    return torch.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)

@torch.no_grad()
def prune_mag_dlp(args, model, tokenizer, device=torch.device("cuda:0"), imp_ratio=None, prune_n=0, prune_m=0):
    time_total = 0

    if imp_ratio is None:
        imp_ratio = get_dlp_ratios(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    layers = model.model.decoder.layers if "opt" in args.model.lower() else model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        time_start = time.time()
        for name in subset:
            layer_sparsity_ratio = 1 - imp_ratio[i]
            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                for ii in range(0, W_metric.shape[1], prune_m):
                    tmp = W_metric[:, ii:ii+prune_m].float()
                    k = min(prune_n, tmp.size(1))  # ← 이 한 줄만 추가
                    if k > 0:
                        idx = torch.topk(tmp, k, dim=1, largest=False).indices
                        W_mask.scatter_(1, ii + idx, True)
            else:
                num_prune = int(W.numel() * layer_sparsity_ratio)
                if num_prune <= 0:
                    W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                elif num_prune >= W.numel():
                    W_mask = torch.ones_like(W_metric, dtype=torch.bool)
                else:
                    flat = W_metric.reshape(-1)
                    # 가장 작은 num_prune번째 값 (정확한 k-th 통계치)
                    kth_val = flat.kthvalue(num_prune).values
                    W_mask = (W_metric <= kth_val)

            # ★ 실제 프루닝 적용(두 분기 공통)
            W[W_mask] = 0

            
        time_end = time.time()
        time_total += time_end - time_start
    print(f"time_total: {time_total}")


def get_dlp_ratios(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):  
    all_layer_ratio=[]
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.decoder.layers if "opt" in args.model.lower() else model.model.layers
    layer_scores = []  # 파이썬 float로만 저장

    for i in range(len(layers)):
        layer = layers[i]
        
        # <<< 레이어 파라미터가 실제 올라간 디바이스로 통일
        layer_dev = next(layer.parameters()).device
        inps  = inps.to(layer_dev, non_blocking=True)
        outs  = outs.to(layer_dev, non_blocking=True)
        attention_mask = _maybe_to(attention_mask, layer_dev)
        position_ids   = _maybe_to(position_ids, layer_dev)

        subset = find_layers(layer)
            
        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in wrapped_layers]

        # 훅 수집용 1회 패스
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    # 주의: 네가 이미 model 인자 추가 버전으로 바꿨다면, 여기 호출도 그에 맞게 바꿔줘.
                    outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)

        for h in handles:
            h.remove()

        # 안전 메트릭 계산
        layer_wmetric = []
        for name in subset:
            W_metric = _safe_w_metric(subset[name].weight.data, wrapped_layers[name].scaler_row)
            # 혹시 모를 dtype 문제 방지
            layer_wmetric.append(W_metric.detach().to("cpu", non_blocking=True).float())

        inps, outs = outs, inps  # 다음 레이어 입력 갱신

        # 레이어 점수(평균) -> 파이썬 float
        if len(layer_wmetric) == 0:
            score = 0.0
        else:
            flat = torch.cat([x.reshape(-1) for x in layer_wmetric], dim=0)
            flat = torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
            score = float(torch.abs(flat.mean()).item())
        layer_scores.append(score)

    # 정규화 구간: 안전 가드
    total_conn = float(sum(layer_scores))
    num_layers = len(layer_scores)

    # 총합이 0이거나 비정상이면 균등 가중치 반환
    if not np.isfinite(total_conn) or total_conn <= 0.0:
        all_layer_ratio = [1.0 - float(args.sparsity_ratio)] * num_layers
        print(all_layer_ratio)
        model.config.use_cache = use_cache
        torch.cuda.empty_cache()
        return all_layer_ratio

    # 1 - (layer / total) 형태의 중요도
    ratio_conn = [1.0 - (s / total_conn) for s in layer_scores]
    imp = torch.tensor(ratio_conn, dtype=torch.float32)
    imp = torch.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)

    # min-max 스케일링 (분모 0 가드)
    mn, mx = float(imp.min().item()), float(imp.max().item())
    denom = mx - mn
    if denom == 0.0:
        scaled = torch.zeros_like(imp)
    else:
        scaled = (imp - mn) * (1.0 / denom * args.alpha * 2.0)

    all_layer = (scaled - scaled.mean() + (1.0 - float(args.sparsity_ratio))).cpu().numpy()
    # 최종 NaN/Inf 정리
    all_layer = np.nan_to_num(all_layer, nan=(1.0 - float(args.sparsity_ratio)),
                              posinf=(1.0 - float(args.sparsity_ratio)),
                              neginf=(1.0 - float(args.sparsity_ratio)))
    all_layer_ratio = all_layer.tolist()

    print(all_layer_ratio)
    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()
    return all_layer_ratio


def prune_wanda_dlp(args, model, tokenizer, device=torch.device("cuda:0"), imp_ratio=None, prune_n=0, prune_m=0):
    time_total = 0

    if imp_ratio is None:
        imp_ratio = get_dlp_ratios(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.decoder.layers if "opt" in args.model.lower() else model.model.layers
    has_map = hasattr(model, "hf_device_map")

    for i in range(len(layers)):
        layer = layers[i]
        
        # <<< 레이어 파라미터가 실제 올라간 디바이스로 통일
        layer_dev = next(layer.parameters()).device
        inps  = inps.to(layer_dev, non_blocking=True)
        outs  = outs.to(layer_dev, non_blocking=True)
        attention_mask = _maybe_to(attention_mask, layer_dev)
        position_ids   = _maybe_to(position_ids, layer_dev)

        subset = find_layers(layer)

        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in wrapped_layers]
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)

        for h in handles:
            h.remove()

        time_start = time.time()
        for name in subset:
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            layer_sparsity_ratio = 1 - imp_ratio[i]
            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask =  torch.zeros_like(W_metric, dtype=torch.bool)
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        k = min(prune_n, tmp.size(1))  # 또는 min(prune_n_i, tmp.size(1))
                        if k > 0:
                            idx = torch.topk(tmp, k, dim=1, largest=False).indices
                            W_mask.scatter_(1, ii + idx, True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                if args.use_variant:
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)
                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - layer_sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha
                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    indices = sort_res[1][:, :int(W_metric.shape[1] * layer_sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0
        time_end = time.time()
        time_total += time_end - time_start

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)

        inps, outs = outs, inps
    print(f"time_total: {time_total}")
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wanda_dlp_structure(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    imp_ratio = get_dlp_structure_ratios(args, model, tokenizer, device, prune_n=0, prune_m=0)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.decoder.layers if "opt" in args.model.lower() else model.model.layers
    has_map = hasattr(model, "hf_device_map")

    for i in range(len(layers)):
        layer = layers[i]
        
        # <<< 레이어 파라미터가 실제 올라간 디바이스로 통일
        layer_dev = next(layer.parameters()).device
        inps  = inps.to(layer_dev, non_blocking=True)
        outs  = outs.to(layer_dev, non_blocking=True)
        attention_mask = _maybe_to(attention_mask, layer_dev)
        position_ids   = _maybe_to(position_ids, layer_dev)

        subset = find_layers(layer)

        prune_n_i = int(imp_ratio[i])
        print('Layer {} prune_n {} prune_m {}'.format(i, prune_n_i, prune_m))

        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in wrapped_layers]

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)

        for h in handles:
            h.remove()

        for name in subset:
            if prune_n_i != 0:
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        k = min(prune_n, tmp.size(1))  # 또는 min(prune_n_i, tmp.size(1))
                        if k > 0:
                            idx = torch.topk(tmp, k, dim=1, largest=False).indices
                            W_mask.scatter_(1, ii + idx, True)
                subset[name].weight.data[W_mask] = 0

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def get_dlp_structure_ratios(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.decoder.layers if "opt" in args.model.lower() else model.model.layers
    has_map = hasattr(model, "hf_device_map")
    layer_score = {}

    for i in range(len(layers)):
        layer = layers[i]
        
        # <<< 레이어 파라미터가 실제 올라간 디바이스로 통일
        layer_dev = next(layer.parameters()).device
        inps  = inps.to(layer_dev, non_blocking=True)
        outs  = outs.to(layer_dev, non_blocking=True)
        attention_mask = _maybe_to(attention_mask, layer_dev)
        position_ids   = _maybe_to(position_ids, layer_dev)

        subset = find_layers(layer)

        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in wrapped_layers]

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)

        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            layer_wmetric.append(W_metric)

        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        if args.strategy == "sum":
            strategy_score = torch.sum(layer_wmetric.float())
        elif args.strategy == "mean":
            strategy_score = torch.mean(layer_wmetric.float())
        elif args.strategy == "max":
            strategy_score = torch.max(layer_wmetric.float())
        elif args.strategy == "median":
            strategy_score = torch.median(layer_wmetric.float())
        elif args.strategy == "std":
            strategy_score = torch.std(layer_wmetric.float())
        elif args.strategy == "var":
            strategy_score = torch.var(layer_wmetric.float())
        per_layer_score = torch.sum(torch.abs(strategy_score))
        layer_score[i] = per_layer_score

    total_conn = sum(layer_score.values())
    ratio_conn = {layer: 1 - (layer_score[layer] / total_conn) for layer in layer_score}

    imp_ratios = torch.tensor(list(ratio_conn.values()))
    min_ratio = torch.min(imp_ratios)
    max_ratio = torch.max(imp_ratios)
    scaled_ratios = (imp_ratios - min_ratio) * (1 / (max_ratio - min_ratio) * args.alpha * 2)
    all_layer_ratio = (scaled_ratios - torch.mean(scaled_ratios) + (1 - args.sparsity_ratio)).tolist()

    all_layer_ratio = np.array(all_layer_ratio)
    all_layer_ratio = np.round(all_layer_ratio)
    all_layer_ratio = prune_n - all_layer_ratio

    print("all_layer_ratio:{}".format(all_layer_ratio))
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    return all_layer_ratio


@torch.no_grad()
def prune_sparsegpt_dlp(args, model, tokenizer, dev, imp_ratio=None, prune_n=0, prune_m=0, mask_recorder: "MaskRecorder" = None):
    time_total = 0

    if imp_ratio is None:
        imp_ratio = get_dlp_ratios(args, model, tokenizer, dev, prune_n=prune_n, prune_m=prune_m)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)

    layers = model.model.decoder.layers if "opt" in args.model.lower() else model.model.layers

    has_map = hasattr(model, "hf_device_map")
    if has_map and ("model.embed_tokens" in model.hf_device_map):
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask', None)
            if "llama" in args.model.lower():
                cache['position_ids'] = kwargs.get('position_ids', None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        layer_sparsity_ratio = 1 - imp_ratio[i]
        if layer_sparsity_ratio <= 0:
            layer_sparsity_ratio = 0.01

        layer = layers[i]
        
        # <<< 레이어 파라미터가 실제 올라간 디바이스로 통일
        layer_dev = next(layer.parameters()).device
        inps  = inps.to(layer_dev, non_blocking=True)
        outs  = outs.to(layer_dev, non_blocking=True)
        attention_mask = _maybe_to(attention_mask, layer_dev)
        position_ids   = _maybe_to(position_ids, layer_dev)

        subset = find_layers(layer)

        gpts = {name: SparseGPT(subset[name]) for name in subset}

        # <<< 프루닝 직전 스냅샷
        before_weights = {name: subset[name].weight.detach().cpu().clone() for name in subset}        

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in gpts]

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)

        for h in handles:
            h.remove()
        time_start = time.time()
        for name in gpts:
            gpts[name].fasterprune(layer_sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        # >>> 프루닝 후 마스크 기록
        if mask_recorder is not None:
            for name in subset:
                after_w = subset[name].weight.data
                mask_recorder.add_from_weights(i, name, before_weights[name], after_w)

        time_end = time.time()
        time_total += time_end - time_start

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)

        layers[i] = layer
        inps, outs = outs, inps
    print(f"time_total: {time_total}")
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    print("model", model)

    if "llama" in args.model.lower():
        layers = model.model.layers
    else:
        layers = model.model.decoder.layers
    print(layers)
    time_total = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        time_start = time.time()
        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W) == 1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        k = min(prune_n, tmp.size(1))  # 또는 min(prune_n_i, tmp.size(1))
                        if k > 0:
                            idx = torch.topk(tmp, k, dim=1, largest=False).indices
                            W_mask.scatter_(1, ii + idx, True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel() * args.sparsity_ratio)].cpu()
                W_mask = (W_metric <= thresh)

            W[W_mask] = 0
        time_end = time.time()
        time_total += time_end - time_start
    print(f"time_total: {time_total}")


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            print('Experiments with OPT models')
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    print("inps", inps)
    layers = model.model.decoder.layers if "opt" in args.model.lower() else model.model.layers
    has_map = hasattr(model, "hf_device_map")

    time_total = 0
    for i in range(len(layers)):
        layer = layers[i]
        # <<< 레이어 파라미터가 실제 올라간 디바이스로 통일
        layer_dev = next(layer.parameters()).device
        inps  = inps.to(layer_dev, non_blocking=True)
        outs  = outs.to(layer_dev, non_blocking=True)
        attention_mask = _maybe_to(attention_mask, layer_dev)
        position_ids   = _maybe_to(position_ids, layer_dev)

        subset = find_layers(layer)

        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in wrapped_layers]
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)

        for h in handles:
            h.remove()
        time_start = time.time()
        for name in subset:
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        k = min(prune_n, tmp.size(1))  # 또는 min(prune_n_i, tmp.size(1))
                        if k > 0:
                            idx = torch.topk(tmp, k, dim=1, largest=False).indices
                            W_mask.scatter_(1, ii + idx, True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0
        time_end = time.time()
        time_total += time_end - time_start
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)

        inps, outs = outs, inps
    print(f"time_total: {time_total}")
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    print('Starting ...')
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.decoder.layers if "opt" in args.model.lower() else model.model.layers

    has_map = hasattr(model, "hf_device_map")
    if has_map and ("model.embed_tokens" in model.hf_device_map):
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask', None)
            if "llama" in args.model.lower():
                cache['position_ids'] = kwargs.get('position_ids', None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')
    time_total = 0
    for i in range(len(layers)):
        layer = layers[i]
        
        # <<< 레이어 파라미터가 실제 올라간 디바이스로 통일
        layer_dev = next(layer.parameters()).device
        inps  = inps.to(layer_dev, non_blocking=True)
        outs  = outs.to(layer_dev, non_blocking=True)
        attention_mask = _maybe_to(attention_mask, layer_dev)
        position_ids   = _maybe_to(position_ids, layer_dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in gpts]

        for j in range(args.nsamples):
            if "llama" in args.model.lower():
                outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)

            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        for h in handles:
            h.remove()
        time_start = time.time()
        for name in gpts:
            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()
        time_end = time.time()
        time_total += time_end - time_start

        for j in range(args.nsamples):
            if "llama" in args.model.lower():
                outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer

        inps, outs = outs, inps
    print(f"time_total: {time_total}")
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def get_owl_ratios(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    all_layer_ratio = []
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.decoder.layers if "opt" in args.model.lower() else model.model.layers
    has_map = hasattr(model, "hf_device_map")

    for i in range(len(layers)):
        layer = layers[i]
        # <<< 레이어 파라미터가 실제 올라간 디바이스로 통일
        layer_dev = next(layer.parameters()).device
        inps  = inps.to(layer_dev, non_blocking=True)
        outs  = outs.to(layer_dev, non_blocking=True)
        attention_mask = _maybe_to(attention_mask, layer_dev)
        position_ids   = _maybe_to(position_ids, layer_dev)

        subset = find_layers(layer)

        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in wrapped_layers]

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)

        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            layer_wmetric.append(W_metric)

        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        for out_ratio in [args.Hyper_m]:
            out_ratio_layer = check_outlier_mean(layer_wmetric, out_ratio)
        all_layer_ratio.append(out_ratio_layer)

    all_layer_ratio = np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1 / (all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda * 2))
    all_layer_ratio = all_layer_ratio - np.mean(all_layer_ratio) + (1 - args.sparsity_ratio)

    print(all_layer_ratio, np.mean(all_layer_ratio), np.max(all_layer_ratio), np.min(all_layer_ratio))
    print("after adjustment", all_layer_ratio)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    return all_layer_ratio


def get_owl_structure_ratios(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    all_layer_ratio = []
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.decoder.layers if "opt" in args.model.lower() else model.model.layers
    has_map = hasattr(model, "hf_device_map")

    for i in range(len(layers)):
        layer = layers[i]
        # <<< 레이어 파라미터가 실제 올라간 디바이스로 통일
        layer_dev = next(layer.parameters()).device
        inps  = inps.to(layer_dev, non_blocking=True)
        outs  = outs.to(layer_dev, non_blocking=True)
        attention_mask = _maybe_to(attention_mask, layer_dev)
        position_ids   = _maybe_to(position_ids, layer_dev)

        subset = find_layers(layer)

        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in wrapped_layers]

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)

        for h in handles:
            h.remove()

        layer_wmetric = []
        for name in subset:
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            layer_wmetric.append(W_metric)

        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        for out_ratio in [args.Hyper_m]:
            out_ratio_layer = check_outlier_mean(layer_wmetric, out_ratio)
            print("layer outlier ratio", out_ratio, out_ratio_layer)
        all_layer_ratio.append(out_ratio_layer)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

    print("before adjustment", all_layer_ratio)

    all_layer_ratio = np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1 / (all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda))
    all_layer_ratio = all_layer_ratio - np.mean(all_layer_ratio)
    all_layer_ratio = np.round(all_layer_ratio)
    all_layer_ratio = prune_n - all_layer_ratio

    print("after adjustment", all_layer_ratio)
    return all_layer_ratio


def prune_wanda_outlier_structure(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    all_layer_ratio = get_owl_structure_ratios(args, model, tokenizer, device=torch.device("cuda:0"))

    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.decoder.layers if "opt" in args.model.lower() else model.model.layers
    has_map = hasattr(model, "hf_device_map")

    for i in range(len(layers)):
        layer = layers[i]
        
        # <<< 레이어 파라미터가 실제 올라간 디바이스로 통일
        layer_dev = next(layer.parameters()).device
        inps  = inps.to(layer_dev, non_blocking=True)
        outs  = outs.to(layer_dev, non_blocking=True)
        attention_mask = _maybe_to(attention_mask, layer_dev)
        position_ids   = _maybe_to(position_ids, layer_dev)

        subset = find_layers(layer)

        prune_n_i = int(all_layer_ratio[i])
        print('Layer {} prune_n {} prune_m {}'.format(i, prune_n_i, prune_m))

        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in wrapped_layers]
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)

        for h in handles:
            h.remove()

        for name in subset:
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

            if prune_n_i != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        k = min(prune_n, tmp.size(1))  # 또는 min(prune_n_i, tmp.size(1))
                        if k > 0:
                            idx = torch.topk(tmp, k, dim=1, largest=False).indices
                            W_mask.scatter_(1, ii + idx, True)
            subset[name].weight.data[W_mask] = 0

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)


        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wanda_outlier(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    all_layer_ratio = get_owl_ratios(args, model, tokenizer, device=torch.device("cuda:0"))
    use_cache = model.config.use_cache
    model.config.use_cache = False
    time_total = 0

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.model.decoder.layers if "opt" in args.model.lower() else model.model.layers
    has_map = hasattr(model, "hf_device_map")

    for i in range(len(layers)):
        layer = layers[i]
        # <<< 레이어 파라미터가 실제 올라간 디바이스로 통일
        layer_dev = next(layer.parameters()).device
        inps  = inps.to(layer_dev, non_blocking=True)
        outs  = outs.to(layer_dev, non_blocking=True)
        attention_mask = _maybe_to(attention_mask, layer_dev)
        position_ids   = _maybe_to(position_ids, layer_dev)

        subset = find_layers(layer)

        wrapped_layers = {name: WrappedGPT(subset[name]) for name in subset}

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in wrapped_layers]
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)

        for h in handles:
            h.remove()

        time_start = time.time()
        for name in subset:
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            layer_sparsity_ratio = 1 - all_layer_ratio[i]
            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)

            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        k = min(prune_n, tmp.size(1))  # 또는 min(prune_n_i, tmp.size(1))
                        if k > 0:
                            idx = torch.topk(tmp, k, dim=1, largest=False).indices
                            W_mask.scatter_(1, ii + idx, True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                if args.use_variant:
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - layer_sparsity_ratio) > 0.001) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > layer_sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha
                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    indices = sort_res[1][:, :int(W_metric.shape[1] * layer_sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
            subset[name].weight.data[W_mask] = 0

        time_end = time.time()
        time_total += time_end - time_start

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)

        inps, outs = outs, inps

    print(f"time_total: {time_total}")
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt_outlier(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    time_total = 0
    all_layer_ratio = get_owl_ratios(args, model, tokenizer, dev)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer)

    layers = model.model.decoder.layers if "opt" in args.model.lower() else model.model.layers

    has_map = hasattr(model, "hf_device_map")
    if has_map and ("model.embed_tokens" in model.hf_device_map):
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask', None)
            if "llama" in args.model.lower():
                cache['position_ids'] = kwargs.get('position_ids', None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        layer_sparsity_ratio = 1 - all_layer_ratio[i]
        if layer_sparsity_ratio <= 0:
            layer_sparsity_ratio = 0.01

        layer = layers[i]
        
        # <<< 레이어 파라미터가 실제 올라간 디바이스로 통일
        layer_dev = next(layer.parameters()).device
        inps  = inps.to(layer_dev, non_blocking=True)
        outs  = outs.to(layer_dev, non_blocking=True)
        attention_mask = _maybe_to(attention_mask, layer_dev)
        position_ids   = _maybe_to(position_ids, layer_dev)

        subset = find_layers(layer)

        gpts = {name: SparseGPT(subset[name]) for name in subset}

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in gpts]

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)

        for h in handles:
            h.remove()
        time_start = time.time()
        for name in gpts:
            gpts[name].fasterprune(layer_sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()
        time_end = time.time()
        time_total += time_end - time_start
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = _layer_forward_llama_safe(layer, model, inps[j].unsqueeze(0), attention_mask, position_ids)


        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps
    print('Total time:', time_total)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_mag_outlier(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    all_layer_ratio = get_owl_ratios(args, model, tokenizer, device)
    time_total = 0

    layers = model.model.decoder.layers if "opt" in args.model.lower() else model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        time_start = time.time()
        for name in subset:
            layer_sparsity_ratio = 1.0 - float(all_layer_ratio[i])
            if layer_sparsity_ratio <= 0:
                layer_sparsity_ratio = 0.01

            W = subset[name].weight.data
            W_metric = torch.abs(W)

            if prune_n != 0:
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                for ii in range(0, W_metric.shape[1], prune_m):
                    tmp = W_metric[:, ii:ii + prune_m].float()
                    k = min(prune_n, tmp.size(1))
                    if k > 0:
                        idx = torch.topk(tmp, k, dim=1, largest=False).indices
                        W_mask.scatter_(1, ii + idx, True)
            else:
                num_prune = int(W.numel() * layer_sparsity_ratio)
                if num_prune <= 0:
                    W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                elif num_prune >= W.numel():
                    W_mask = torch.ones_like(W_metric, dtype=torch.bool)
                else:
                    flat = W_metric.reshape(-1)
                    kth_val = flat.kthvalue(num_prune).values
                    W_mask = (W_metric <= kth_val)
            W[W_mask] = 0

        time_end = time.time()
        time_total += time_end - time_start
    print(f"total time {time_total}")


