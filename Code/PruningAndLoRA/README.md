# Progressive Serve 실험 재현 가이드

## 개요
PruningAndLoRA는 Llama2-7B 모델을 대상으로 
- 각도 기반 연속 레이어 프루닝
- 프루닝 후 레이어 그룹 A, 나머지 레이어들은 레이어 그룹 B,C로 나누어 저장
- A lora 어댑터/AB lora 어댑터 생성 
을 수행하는 실험 코드입니다.
즉, 프루닝, LoRA 어댑터 생성을 담당하는 부분입니다.

## 파일 구성

```
PruningAndLoRA/
├── lib/
│   ├── __init__.py
│   ├── bundler.py          # 제거된 레이어를 B/C 그룹으로 분리 저장
│   ├── data.py             # 프루닝 중요도 계산, calibration, LoRA 학습을 위한 언어모델 입력 시퀀스 생성 유틸리티
│   ├── identity.py         # 프루닝으로 제거된 레이어 자리를 대신하는 PassLayer (구조 보존용)
│   ├── layeronly_drop.py   # 프루닝 실행 코드
│   └── simdrop.py          # Angular-distance 기반 연속 레이어 프루닝
│
├── total_progressive_qa_lora.py  # SQuAD 기반 LoRA 어댑터 학습
├── pruningandlora.md             # 실험 설명 문서 (ReadME)
└── requirements.txt              # Python 의존성

```

## 실행 전 준비

### 1. venv 환경 활성화
앞에서 이 과정을 진행했을 경우 생략
```bash
# venv 환경 생성 (최초 1회)
python -m venv .venv

# 가상환경 활성화
source .venv/bin/activate
```


### 2. 필수 패키지 설치
가상환경 활성화 후 필수 패키지를 설치하여야 합니다.
```bash
# 필수 패키지 설치
pip install -r requirements.txt

# torch는 따로 추가적으로 설치해줍니다.
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121

```
### 3. Python 코드 내 경로 확인
현재 경로가 ./growth인지 확인합니다. 경로가 맞지 않으면 코드가 정상적으로 실행되지 않습니다. 


## 실험 실행 순서

### Stage 1: 프루닝
```
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
- 모델 프루닝 실행
- 결과: ./Code/results/pruning/A (레이어그룹 A), ./Code/results/pruning/bundles(레이어그룹B,레이어그룹C 저장) 

### Stage 2: LoRA 어댑터 생성
```
# Stage1
python -m Code.PruningAndLoRA.total_progressive_qa_lora \
  --base_dir ./results/pruning/A \
  --bundles_dir ./results/pruning/bundles \
  --stage 1 \
  --out_adapters ./results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 --seq_len 512 --epochs 1 --bs 4 --grad_acc 8
 

# Stage2
python Code.PruningAndLoRA.total_progressive_qa_lora.py \
  --base_dir ~/Code/results/pruning/A \
  --bundles_dir ~/Code/results/pruning/bundles \
  --stage 2 \
  --out_adapters ~/Code/results/adapters \
  --qa_dataset squad --max_samples 20000 --max_eval_samples 8000 --seq_len 512 --epochs 1 --bs 4 --grad_acc 8

```
- 레이어 그룹 A에 대한 A lora 어댑터 생성
- 레이어 그룹 A + 레이어 그룹 B에 대한 AB lora 어댑터 생성
- 결과: ./Code/results/adapters (A 어댑터, AB 어댑터 저장)



