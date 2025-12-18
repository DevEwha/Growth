# ProgressiveServe: Pruning + LoRA + Serving 재현 가이드

본 문서는 **PruningAndLoRA → Evaluation → ProgressiveServe(Serving)** 전체 파이프라인을 처음 보는 사람도 그대로 따라 실행하면 **논문 실험이 재현 가능**하도록 작성된 통합 README입니다.

모델 및 LoRA 어댑터는 Hugging Face에 **이미 준비된 결과물**을 제공하므로, *프루닝/LoRA 학습을 직접 다시 돌리지 않고도* 평가 및 서빙 실험을 재현할 수 있습니다.

---

## 0. 전체 구조 한눈에 보기

```
Growth/
├── Code/
│   ├── PruningAndLoRA/     # (선택) 프루닝 + LoRA 생성 코드
│   ├── Evaluation/         # TriviaQA 평가 코드
│   └── Serving/            # ProgressiveServe 서빙 코드
└── README.md               # (본 문서)
```

### 재현 방법 요약

* **빠른 재현(권장)**: Hugging Face에서 모델 다운로드 → Evaluation + Serving 실행
* **완전 재현**: Pruning → LoRA 학습 → Evaluation → Serving

---

## 1. 사전 준비

### 1.1 시스템 요구사항

* OS: Linux
* GPU: NVIDIA GPU (CUDA 지원)
* CUDA: 12.x 권장
* Python: 3.9 이상

---

## 2. (권장) Hugging Face에서 준비된 모델 받기

본 실험에서 사용하는 **Stage 1 / Stage 2 / Stage 3 모델과 LoRA 어댑터는 아래 Hugging Face 리포지토리에 모두 업로드되어 있습니다.**

🔗 **Model & Adapter Repository**
[https://huggingface.co/dddreamerrr/pruning_lora_results](https://huggingface.co/dddreamerrr/pruning_lora_results)

### 2.1 다운로드 방법

```bash
# 원하는 위치에서
mkdir models && cd models

git lfs install
git clone https://huggingface.co/dddreamerrr/pruning_lora_results
```

다운로드 후 구조 예시:

```
models/pruning_lora_results/
├── stage1/
├── stage2/
├── stage3/
└── adapters/
```

> ⚠️ **중요**: 이후 모든 Evaluation / Serving 코드에서 이 경로를 `base_dir`로 사용합니다.

---

## 3. (선택) Pruning + LoRA 생성 전체 재현

> ⏱️ **시간이 오래 걸리므로 논문 재현 목적이라면 생략 가능**

### 3.1 환경 설정

```bash
cd Code/PruningAndLoRA
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

### 3.2 프루닝 실행 (Stage 1)

```bash
python -m Code.PruningAndLoRA.lib.layeronly_drop \
  --model meta-llama/Llama-2-7b-hf \
  --device cuda:0 \
  --drop_frac 0.25 \
  --keep_last_layer \
  --nsamples 64 \
  --seqlen 2048 \
  --max_batches 32 \
  --save_dir ./results/pruning/A \
  --save_removed_dir ./results/pruning/bundles
```

### 3.3 LoRA 어댑터 생성

```bash
# Stage 1
python Code.PruningAndLoRA.total_progressive_qa_lora.py \
  --base_dir ./results/pruning/A \
  --bundles_dir ./results/pruning/bundles \
  --stage 1 \
  --out_adapters ./results/adapters \
  --qa_dataset squad --epochs 1

# Stage 2
python Code.PruningAndLoRA.total_progressive_qa_lora.py \
  --base_dir ./results/pruning/A \
  --bundles_dir ./results/pruning/bundles \
  --stage 2 \
  --out_adapters ./results/adapters \
  --qa_dataset squad --epochs 1
```

---


## 4. Evaluation (TriviaQA Zero-shot 평가)

### 4.1 Conda 환경 설정

```bash
conda env create -f environment.yml
conda activate sllm_exp
```

### 4.2 경로 설정 (중요)

`j_eval_*.py` 파일 내부의 `Config` 클래스에서 **모델 경로를 수정**합니다.

```python
@dataclass
class Config:
    base_dir: str = "/ABSOLUTE/PATH/models/pruning_lora_results"
    device: str = "cuda:0"
```

### 4.3 실행 권한 부여

```bash
chmod +x j_shell_*.sh
```

### 4.4 단계별 평가 실행

#### Stage 0: Origin

```bash
bash j_shell_origin_TriviaQA.sh
```

#### Stage 1

```bash
bash j_shell_newstage1_TriviaQA.sh
```

#### Stage 2

```bash
bash j_shell_newstage2_TriviaQA.sh
```

#### Stage 3

```bash
bash j_shell_newstage3_TriviaQA.sh
```

### 4.5 결과 확인

각 실행 후 CSV가 생성됩니다.

```
*_triviaqa_eval_900.csv
```

포함 지표:

* **Exact Match (EM)**
* **F1 Score**


---

## 5. ProgressiveServe (점진적 로딩 서빙 실험)

### 5.1 환경 설정

```bash
cd Code/Serving
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 5.2 실행

```bash
python progressive_serve.py 
```

---
### 6. 참고
- 구체적인 방법은 각 실험의 폴더 설명을 참고
- 정확한 실험 재현을 위해서는 원격 서버와 모델 서버를 따로 두어 원격 서버에 모델을 다운, 모델 서버에서 Fetch를 해야 함