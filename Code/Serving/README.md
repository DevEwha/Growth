# ProgressiveServe 서빙 실험 재현 가이드

## 개요
ProgressiveServe의 PruningAndLoRA를 통해 준비된 모델 및 어댑터를 통해 점진적 로딩에 대한 응답을 받아보는 실험입니다.<br>
Llama2-7B 모델에 대해 기본적으로 평가하도록 코드가 구성되어 있습니다.

## 파일 구성
```
Serving/
├── model_utils.py        # progressive_serve 핵심 함수
├── progressive_serve.py  # 점진적 로딩 및 응답
├── pull.py               # HuggingFace 모델을 받아오는 코드
└── requirements.txt      # 필요 환경
```

## 설치 방법
```
git clone https://github.com/DevEwha/Growth.git
cd Growth/Code/Serving
pip install -r requirements.txt
```

## 실험 실행

### 1. 가상환경 활성화
앞에서 이 과정을 진행했을 경우 생략합니다.
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
source venv/bin/activate

# requirements.txt 설치
pip install -r requirements.txt
```

### 2. 모델 다운로드
PruningAndLoRA를 통해 이미 모델이 준비되었을 경우 생략합니다.
```bash
python pull.py
```

### 3. 코드 실행
```bash
python progressive_serve.py
```

## 주의사항
- GPU 메모리가 부족하면 `CUDA_VISIBLE_DEVICES` 환경 변수로 GPU 조정합니다.
