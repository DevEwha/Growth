# ProgressiveServe: 서버리스 LLM 콜드 스타트 완화를 위한 점진적 모델 로딩 및 복구

**서버리스 환경에서 LLM 콜드 스타트를 줄이기 위한 프루닝·LoRA·점진적 로딩 파이프라인**

## 📖 개요 (Overview)

ProgressiveServe는 서버리스 환경에서 대규모 언어 모델(LLM)을 서빙할 때 발생하는 심각한 콜드 스타트 지연을 줄이기 위한 연구용 프로토타입 시스템입니다. 프루닝된 경량 모델을 먼저 로딩·서빙하고, 백그라운드에서 전체 모델을 점진적으로 복구함으로써 초기 응답 시간을 줄이면서 최종 정확도는 원본 모델 수준으로 유지합니다.

### 주요 아이디어

- **각도 기반 연속 레이어 프루닝**: 레이어 입력·출력 간 코사인 유사도를 계산하여 중요도가 낮은 레이어를 선택적으로 제거하여 모델을 경량화합니다.
- **3단계 점진적 복구**:
  - 단계 1: 레이어 그룹 A만 로딩한 경량 모델 서빙  
  - 단계 2: 레이어 그룹 B를 추가 로딩하여 모델 품질 향상  
  - 단계 3: 레이어 그룹 C까지 로딩하여 원본 모델과 동일한 구조로 복구
- **LoRA 어댑터를 이용한 성능 복구**: 단계별로 서로 다른 LoRA 어댑터(A, AB)를 부착하여 프루닝으로 인한 성능 저하를 완화합니다.
- **PassLayer 메커니즘**: 아직 로딩되지 않은 레이어 위치를 플레이스홀더 레이어로 채워 서비스 중단 없이 실제 레이어로 교체할 수 있도록 합니다.

### 성능 요약

TriviaQA 검증 세트와 Llama2-7B 기준 실험 결과는 다음과 같습니다.

| 방법              | TTFT (s) | EM (%) | F1 (%) |
|------------------|----------|--------|--------|
| ServerlessLLM | 114      | 55.67  | 66.11  |
| ProgressiveServe 단계 1 | 90       | 48.22  | 54.26  |
| ProgressiveServe 최종 단계 | 90       | 55.67  | 66.11  |

ProgressiveServe는 선행 연구 대비 TTFT를 약 21.1% 단축하면서 최종 단계에서는 EM/F1이 원본과 동일한 수준에 도달합니다.

***

## 🗂️ 프로젝트 구조 (Source Code 설명)

리포지토리는 대략 다음과 같은 구조로 구성되어 있습니다.

```text
Growth/
├── 1stReport/                  # 1차 보고서 자료
├── 2ndReport/                  # 2차 보고서 자료
├── Code/                       # 실험 및 서빙 관련 코드
│   ├── Check/                  # 모델 성능 평가 및 검증 코드
│   │   ├── environment.yml
│   │   ├── j_eval_newstage1_fixed_TriviaQA.py
│   │   ├── j_eval_newstage2_fixed_TriviaQA.py
│   │   ├── j_eval_newstage3_fixed_TriviaQA.py
│   │   ├── j_eval_origin3_TriviaQA.py
│   │   ├── j_shell_newstage1_TriviaQA.sh
│   │   ├── j_shell_newstage2_TriviaQA.sh
│   │   ├── j_shell_newstage3_TriviaQA.sh
│   │   ├── j_shell_origin_TriviaQA.sh
│   │   ├── log.py
│   │   ├── model_utils.py
│   │   ├── logs/               # 실행 로그 저장
│   │   ├── result/             # 평가 결과 저장
│   │   ├── __pycache__/
│   │   └── README.md
│   │
│   ├── PruningAndLoRA/          # Pruning 및 LoRA 기반 실험 코드
│   │   ├── lib/                 # 공용 라이브러리
│   │   ├── total_progressive_qa_lora.py
│   │   ├── pruningandlora.md
│   │   ├── requirements.txt
│   │   ├── README.md
│   │   └── __pycache__/
│   │
│   ├── Serving/                 # 모델 서빙 관련 코드
│   │   ├── models/              # 서빙용 모델 파일
│   │   ├── progressive_serve.py
│   │   ├── model_utils.py
│   │   ├── pull.py
│   │   ├── requirements.txt
│   │   ├── README.md
│   │   ├── venv/                # 가상환경
│   │   └── __pycache__/
│   │
│   ├── drop_run.log
│   └── README.md
│
├── results/                     # 실험 결과 정리
├── drop_run.log
├── .gitignore
└── README.md
```

***

## 🔧 설치 방법 (How to install)

### 1. 환경 요구사항

* OS: Linux
* GPU: NVIDIA GPU (CUDA 지원) (RTX 3090/4090 급 24GB VRAM 이상)
* CUDA: 12.x 권장
* Python: 3.9 이상

### 2. 리포지토리 클론

```bash
# 리포지토리 클론
git clone https://github.com/DevEwha/Growth.git
cd Growth
```

***

## 🚀 실행 방법 (How to run / How to test)
본 문서는 **PruningAndLoRA → Evaluation → ProgressiveServe(Serving)** 전체 파이프라인을 처음 보는 사람도 그대로 따라 실행하면 **논문 실험이 재현 가능**하도록 작성된 통합 README입니다.

모델 및 LoRA 어댑터는 Hugging Face에 **이미 준비된 결과물**을 제공하므로, *프루닝/LoRA 학습을 직접 다시 돌리지 않고도* 평가 및 서빙 실험을 재현할 수 있습니다.

### 재현 방법 요약

* **빠른 재현(권장)**: Hugging Face에서 모델 다운로드 → Evaluation + Serving 실행
* **완전 재현**: Pruning → LoRA 학습 → Evaluation → Serving

### 1. 사전 준비

#### 1.1 시스템 요구사항

* OS: Linux
* GPU: NVIDIA GPU (CUDA 지원) (RTX 3090/4090 급 24GB VRAM 이상)
* CUDA: 12.x 권장
* Python: 3.9 이상


### 2. (권장) Hugging Face에서 준비된 모델 받기

본 실험에서 사용하는 **Stage 1 / Stage 2 / Stage 3 모델과 LoRA 어댑터는 아래 Hugging Face 리포지토리에 모두 업로드되어 있습니다.**

🔗 **Model & Adapter Repository**
[https://huggingface.co/dddreamerrr/pruning_lora_results](https://huggingface.co/dddreamerrr/pruning_lora_results)

#### 2.1 다운로드 방법

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
models/pruning_lora_results/
├── A/                # Stage 1: Pruned model (Layer group A only)
├── adapters/         # LoRA adapters
│   ├── A_lora/       # LoRA trained on A layers (Stage 1)
│   └── AB_lora/      # LoRA trained on A+B layers (Stage 2)
└── bundles/          # Removed layers stored for recovery
   ├── B/            # Layer group B
   └── C/            # Layer group C

```

> ⚠️ **중요**: 이후 모든 Evaluation / Serving 코드에서 이 경로를 `base_dir`로 사용합니다.


### 3. (선택) Pruning + LoRA 생성 전체 재현

> ⏱️ **시간이 오래 걸리므로 논문 재현 목적이라면 생략 가능**

#### 3.1 환경 설정

```bash
cd Code/PruningAndLoRA
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

#### 3.2 프루닝 실행 (Stage 1)

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

#### 3.3 LoRA 어댑터 생성

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

### 4. Evaluation (TriviaQA Zero-shot 평가)

#### 4.1 Conda 환경 설정

```bash
conda env create -f environment.yml
conda activate sllm_exp
```

#### 4.2 경로 설정 (중요)

`j_eval_*.py` 파일 내부의 `Config` 클래스에서 **모델 경로를 수정**합니다.

```python
@dataclass
class Config:
    base_dir: str = "/ABSOLUTE/PATH/models/pruning_lora_results"
    device: str = "cuda:0"
```

#### 4.3 실행 권한 부여

```bash
chmod +x j_shell_*.sh
```

#### 4.4 단계별 평가 실행

* Stage 0: Origin

```bash
bash j_shell_origin_TriviaQA.sh
```

* Stage 1

```bash
bash j_shell_newstage1_TriviaQA.sh
```

* Stage 2

```bash
bash j_shell_newstage2_TriviaQA.sh
```

* Stage 3

```bash
bash j_shell_newstage3_TriviaQA.sh
```

#### 4.5 결과 확인

각 실행 후 CSV가 생성됩니다.

```
*_triviaqa_eval_900.csv
```

포함 지표:

* **Exact Match (EM)**
* **F1 Score**


### 5. ProgressiveServe (점진적 로딩 서빙 실험)

#### 5.1 환경 설정

```bash
cd Code/Serving
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 5.2 실행

```bash
python progressive_serve.py 
```

#### 6. 참고
- 구체적인 방법은 각 실험의 폴더 설명을 참고
- 정확한 실험 재현을 위해서는 원격 서버와 모델 서버를 따로 두어 원격 서버에 모델을 다운, 모델 서버에서 Fetch를 해야 함

***

## 🧪 실험 데이터 및 결과

### 실험 데이터
- TriviaQA 검증 샘플 일부(예: 100개)를 포함하며 EM/F1 평가에 사용됩니다.
- LoRA 어댑터 학습에 사용되는 SQuAD 학습 샘플을 포함합니다.

TriviaQA 평가 설정은 zero-shot, max_new_tokens=10, greedy decoding으로 고정하여 단계별 성능을 비교합니다.

### 실험 결과물

- https://huggingface.co/dddreamerrr/pruning_lora_results

***

## 📚 사용한 데이터/오픈소스 정리

### 사용 데이터셋

- **SQuAD**: LoRA 어댑터 학습용 QA 데이터셋
- **TriviaQA**: 단계별 EM/F1 평가용 QA 데이터셋

### 사용 오픈소스

| 라이브러리     | 용도                           | 라이선스   |
|----------------|--------------------------------|-----------|
| PyTorch        | 딥러닝 프레임워크              | BSD-3-Clause |
| Hugging Face Transformers | Llama2-7B 로딩 및 토크나이저 | Apache 2.0 |
| PEFT           | LoRA 어댑터 구현 및 학습       | Apache 2.0 |
| Ray Serve      | 서버리스 유사 서빙 인프라      | Apache 2.0 |
| Safetensors    | 모델 체크포인트 저장 포맷      | Apache 2.0 |
| Datasets       | SQuAD, TriviaQA 로딩           | Apache 2.0 |

### 주요 참고 문헌

1. **ServerlessLLM**: Y. Fu et al., "ServerlessLLM: Low-latency serverless inference for large language models," USENIX OSDI 2024
2. **LoRA**: E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," arXiv:2106.09685
3. **Layer Pruning**: A. Gromov et al., "The unreasonable ineffectiveness of the deeper layers," arXiv:2403.17887
4. **Llama 2**: H. Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models," arXiv:2307.09288

***

## 👩‍💻 저자 및 연락처

**공동 제1저자**

- 박나담 (Nadam Park) – parknd@ewhain.net  
- 이나경 (Nakyeong Lee) – rinarina0429@ewha.ac.kr  
- 이주원 (Juwon Lee) – juwonlee.cse@gmail.com  

**지도교수**

- 심재형 (Jaehyeong Sim) – jh.sim@ewha.ac.kr  

이화여자대학교 컴퓨터공학과

***

## 📝 라이선스

본 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고해주세요.
