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

- OS: Ubuntu 24.04 (또는 유사 Linux 환경)
- GPU: NVIDIA RTX 계열 (논문 실험은 RTX 5090 사용)
- 드라이버: NVIDIA Driver 580.65.06+ 및 CUDA 13.0+
- Python 3.9 이상  
- 원격 모델 스토리지를 위한 NFSv4 (16Gbps 이더넷 환경에서 테스트)

### 2. 리포지토리 클론

```bash
# 리포지토리 클론
git clone https://github.com/DevEwha/Growth.git
cd Growth
```

***

## 🏗️ 빌드 및 준비 (How to build)

### 1. 프루닝된 레이어 그룹 생성 (오프라인 단계)

```bash
bash scripts/prepare_pruned_models.sh \
  --model_path models/llama2-7b-base \
  --output_dir models/llama2-7b-pruned \
  --prune_layers 21-28
```

이 스크립트는 논문에서 제안한 각도 기반 연속 레이어 프루닝 기법을 이용하여 32개 레이어 중 21–28번 레이어를 제거하고,  
- 그룹 A: 1–20, 29–32  
- 그룹 B: 21–24  
- 그룹 C: 25–28  
로 분리·저장합니다.

### 2. LoRA 어댑터 학습

```bash
bash scripts/train_lora_adapters.sh \
  --base_model models/llama2-7b-base \
  --pruned_groups models/llama2-7b-pruned \
  --dataset experiments/data/squad_train.json \
  --output_dir models/lora_adapters
```

- A 어댑터: 단계 1에서 그룹 A 위주로 성능 복구
- AB 어댑터: 단계 2에서 그룹 A+B에 대해 최적화

학습에는 대표적인 QA 데이터셋 SQuAD가 사용됩니다.

***

## 🚀 실행 방법 (How to run / How to test)

### 1. ProgressiveServe 테스트

#### 1. 가상환경 활성화
앞에서 이 과정을 진행했을 경우 생략합니다.
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
source venv/bin/activate

# requirements.txt 설치
pip install -r requirements.txt
```

#### 2. 모델 다운로드
PruningAndLoRA를 통해 이미 모델이 준비되었을 경우 생략합니다.
```bash
python pull.py
```

#### 3. 코드 실행
```bash
python progressive_serve.py
```
이 스크립트는 단계 1 → 2 → 3 순서로 레이어를 로딩하면서 동일 세션 내에서 모델 구조를 점진적으로 복구합니다.

### 2. Ray Serve 기반 "서버리스" 시나리오 실행

```bash
# Ray Serve로 배포 (콜드 스타트 측정을 위해 매 요청 시 Actor 새로 생성하도록 설정)
bash scripts/deploy_serve.sh \
  --nfs_mount /mnt/nfs_models \
  --gpu_count 1 \
  --port 8000
```

```bash
# HTTP 요청 예시
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is ProgressiveServe?", "max_tokens": 50}'
```

논문에서는 매 트라이얼마다 Ray Actor를 새로 만들고 OS 페이지 캐시와 GPU 캐시를 초기화하여 TTFT를 측정하였습니다.[1]

***

## 🧪 실험 데이터 및 결과

### 실험 데이터

- `experiments/data/triviaqa_samples.json`  
  - TriviaQA 검증 샘플 일부(예: 100개)를 포함하며 EM/F1 평가에 사용됩니다.
- `experiments/data/squad_train.json`  
  - LoRA 어댑터 학습에 사용되는 SQuAD 학습 샘플을 포함합니다.

TriviaQA 평가 설정은 zero-shot, max_new_tokens=10, greedy decoding으로 고정하여 단계별 성능을 비교합니다.[1]

### 실험 결과물

- `experiments/results/cold_start_metrics.csv`  
  - 단계별 TTFT 및 전체 벽시계 시간 측정값이 포함되어 있습니다.[1]
- `experiments/results/qa_performance.csv`  
  - ServerlessLLM vs ProgressiveServe(단계 1/2/3)의 EM, F1 점수가 기록됩니다.[1]
- `experiments/results/ablation_study.csv`  
  - LoRA 유무, 단계별 구성 등 요소별 성능 기여도를 분석한 결과가 포함됩니다.[1]

### 실험 재현 방법

```bash
# 전체 평가 파이프라인 실행 (TriviaQA zero-shot)
python experiments/evaluate_cold_start.py \
  --dataset experiments/data/triviaqa_samples.json \
  --baseline serverlessllm \
  --proposed progressiveserve \
  --trials 10 \
  --clear_cache

# 결과는 experiments/results/ 디렉토리에 저장됩니다
```

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

**공동 제1저자***

- 박나담 (Nadam Park) – parknd@ewhain.net  
- 이나경 (Nakyeong Lee) – rinarina0429@ewha.ac.kr  
- 이주원 (Juwon Lee) – juwonlee.cse@gmail.com  

**지도교수**

- 심재형 (Jaehyeong Sim) – jh.sim@ewha.ac.kr  

이화여자대학교 컴퓨터공학과

***

## 📝 라이선스

본 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고해주세요.
