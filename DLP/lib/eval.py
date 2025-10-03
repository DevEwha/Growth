# Import necessary modules
import torch
import torch.nn as nn
import sys
# Import get_loaders function from data module within the same directory
from .data import get_loaders
import fnmatch
from pdb import set_trace as st

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(model, tokenizer, device=torch.device("cuda:0"), dataset="wikitext2"):

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl = eval_ppl_dataset(model, testloader, 1, device)
    return ppl 

""" # Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_dataset(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # st()
        
        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to("cuda:1")
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)


        # print ("nlls",nlls)
        sys.stdout.flush()

    
    print ('begin calcualte ppl')
    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    # torch.cuda.empty_cache()

    return ppl.item()"""
# lib/eval.py 안의 eval_ppl_dataset를 이 버전으로 교체

def eval_ppl_dataset(model, testenc, bs=1, device=None):
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    nlls = []
    tok_count = 0
    print(f"nsamples {nsamples}")

    loss_fct = nn.CrossEntropyLoss()

    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")
        j = min(i + bs, nsamples)

        inputs = testenc[:, (i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j - i, model.seqlen)
        attn_mask = torch.ones_like(inputs, device=device)

        with torch.no_grad():
            out = model(inputs, attention_mask=attn_mask).logits
            # 안정화: fp32로 올리고 NaN/Inf 정화
            logits = torch.nan_to_num(out.float(), nan=0.0, posinf=1e4, neginf=-1e4)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:].contiguous()

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        if not torch.isfinite(loss):
            print(f"[warn] non-finite loss at batch {i}: {loss.item()}; skipping this batch")
            continue

        # 유효 토큰 수만큼 누적
        nlls.append(loss * (shift_labels.numel()))
        tok_count += shift_labels.numel()

        sys.stdout.flush()

    print('begin calcualte ppl')
    if tok_count == 0:
        return float('nan')  # 모든 배치가 NaN이면 그대로 반환

    ppl = torch.exp(torch.stack(nlls).sum() / tok_count)
    return ppl.item()




""" 
def eval_zero_shot(model_name, task_list=["arc_easy","arc_challenge","hellaswag","boolq","piqa","winogrande","lambada_openai"],
        num_fewshot=0, use_accelerate=True, add_special_tokens=False, use_base_tokenizer=False, batch_size=4):
    from lm_eval import tasks, evaluator 
    
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    #model_args = f"pretrained={model_name},cache_dir=./llm_weights"

    #if use_accelerate:
    #   model_args = f"pretrained={model_name},use_accelerate=True,device_map_option=\"auto\""
    # 핵심: fp16 + device_map, (옵션) 토크나이저 원본 경로
    model_args = f"pretrained={model_name},dtype=float16,device_map_option=auto"
    if use_base_tokenizer:
        model_args += ",tokenizer=meta-llama/Llama-2-7b-hf"

    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        max_batch_size=None,
        device=None,
        no_cache=True,
        # limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        write_out=False,
        output_base_path=None
    )
    print("********************************")
    print("zero_shot evaluation results")
    print(evaluator.make_table(results))
    # st()
    return results  """

def eval_zero_shot(
    model_name,
    task_list,
    num_fewshot=0,
    tokenizer=None,          # 예: "meta-llama/Llama-2-7b-hf"
    dtype="float16",
    use_accelerate=True,
    device_map_option="auto",
    batch_size=4,
):
    import fnmatch
    from lm_eval import evaluator

    # 1) 태스크 목록 얻기(버전 호환)
    ALL = None
    try:
        from lm_eval import list_tasks
        ALL = list_tasks()
    except Exception:
        try:
            from lm_eval import tasks
            ALL = getattr(tasks, "ALL_TASKS", None)
        except Exception:
            pass
    if ALL is None:
        try:
            from lm_eval.api.registry import TaskManager
            tm = TaskManager()
            if hasattr(tm, "get_task_dict"):
                ALL = list(tm.get_task_dict().keys())
            elif hasattr(tm, "task_name_to_cls"):
                ALL = list(tm.task_name_to_cls.keys())
        except Exception:
            ALL = None

    def pattern_match(patterns, source_list):
        if source_list is None:
            return list(patterns)
        keep = set()
        for pat in patterns:
            for name in source_list:
                if fnmatch.fnmatch(name, pat):
                    keep.add(name)
        return sorted(keep)

    patterns = task_list if isinstance(task_list, (list, tuple)) else [task_list]
    task_names = pattern_match(patterns, ALL)

    # 2) 사용할 모델 백엔드 자동 선택
    from lm_eval.api.registry import MODEL_REGISTRY
    backends_pref = [
        "hf-causal-experimental",  # 최신
        "hf-auto",                 # 중간
        "hf", "huggingface"        # 구버전
    ]
    model_backend = None
    for b in backends_pref:
        if b in MODEL_REGISTRY:
            model_backend = b
            break
    if model_backend is None:
        # 레지스트리에 보이는 백엔드 목록 참고용 출력
        print("[warn] supported model backends:", sorted(MODEL_REGISTRY.keys()))
        raise ValueError("No compatible HF backend found in lm_eval registry.")

    # 3) model_args 구성 (백엔드별로 안전한 옵션만)
    args = [f"pretrained={model_name}"]
    if tokenizer:
        args.append(f"tokenizer={tokenizer}")
    if dtype:
        args.append(f"dtype={dtype}")
    # 가급적 최신/중간 백엔드에서만 가속 옵션 전달
    if model_backend in ("hf-causal-experimental", "hf-auto"):
        if use_accelerate:
            args.append("use_accelerate=True")
        if device_map_option:
            args.append(f"device_map_option={device_map_option}")

    model_args = ",".join(args)

    # 4) 구버전이 싫어하는 키워드 제거하면서 재시도하는 래퍼
    import re
    def simple_eval_compat(**kw):
        try:
            return evaluator.simple_evaluate(**kw)
        except TypeError as e:
            m = re.search(r"unexpected keyword argument '([^']+)'", str(e))
            if not m:
                raise
            bad = m.group(1)
            kw.pop(bad, None)
            return simple_eval_compat(**kw)

    # 5) 최소 안전 인자만 전달 (구버전 호환)
    results = simple_eval_compat(
        model=model_backend,
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device="cuda:0",   # 구버전은 여기서 디바이스 지정
    )

    print("********************************")
    print("zero_shot evaluation results")
    print(evaluator.make_table(results))
    return results
