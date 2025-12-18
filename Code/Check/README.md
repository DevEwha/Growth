# Progressive Serve 실험 재현 가이드

## 개요
TriviaQA 데이터셋으로 progressive pruning + LoRA adaptation 성능을 평가하는 실험입니다.  
Origin(기본 모델)부터 Stage1, Stage2, Stage3까지 단계적으로 실행하며 성능 변화를 측정합니다.

## 파일 구성

```
├── environment.yml                     # conda 환경을 만들기 위한 yaml 파일
├── j_eval_origin3_TriviaQA.py          # Origin 모델 평가
├── j_eval_newstage1_fixed_TriviaQA.py  # Stage 1 평가
├── j_eval_newstage2_fixed_TriviaQA.py  # Stage 2 평가
├── j_eval_newstage3_fixed_TriviaQA.py  # Stage 3 평가
├── j_shell_origin_TriviaQA.sh          # Origin 실행 스크립트
├── j_shell_newstage1_TriviaQA.sh       # Stage 1 실행 스크립트
├── j_shell_newstage2_TriviaQA.sh       # Stage 2 실행 스크립트
├── j_shell_newstage3_TriviaQA.sh       # Stage 3 실행 스크립트
├── log.py                              # 로그 작성을 위한 함수 모음
├── model_utils.py                      # Porgressive Load 핵심 함수
└── results                             # 참고를 위한 결과 데이터 첨부
```

## 실행 전 준비

### 1. Conda 환경 활성화
앞에서 이 과정을 진행했을 경우 생략
```bash
# Conda 환경 생성 (최초 1회)
conda env create -f environmental.yml

# Conda 환경 활성화
conda activate sllm_exp
```


### 2. Python 코드 내 경로 수정
각 Python 파일(`j_eval_*.py`)에서 `Config` 클래스의 `base_dir`를 실제 모델 경로로 수정
모델 다운 받는 법은 앞에서 개괄 설명한 md에서 참고(상위 Code 디렉토리 위치에 있는 md 참고)

```
@dataclass
class Config:
    base_dir: str = "/your/actual/path/to/models"  # 이 부분을 실제 모델이 있는 폴더 위치로 수정해 주세요
    device: str = "cuda:0"
```

### 3. Bash 스크립트 실행 권한 부여
```
chmod +x j_shell_*.sh
```

## 실험 실행 순서

### Stage 0: Origin (baseline)
```
bash j_shell_origin_TriviaQA.sh
```
- 원본 모델의 베이스라인 성능 측정
- 결과: `origin_triviaqa_eval_900.csv`

### Stage 1: 1차 Pruning모델(A레이어그룹) + LoRA(A레이어그룹 전용)
```
bash j_shell_newstage1_TriviaQA.sh
```
- 1차 레이어 프루닝 후 LoRA 적용
- 결과: `stage1_triviaqa_eval_900.csv`

### Stage 2: 2차 Pruning모델(A레이어그룹)+B레이어 그룹 + LoRA(AB레이어그룹 전용)
```
./j_shell_newstage2_TriviaQA.sh
```
- 2차 레이어 프루닝 후 LoRA 적용
- 결과: `stage2_triviaqa_eval_900.csv`

### Stage 3: 3차 전체 모델
```
./j_shell_newstage3_TriviaQA.sh
```
- 전체 레이어가 존재하는 모델로 복구 완료
- 결과: `stage3_triviaqa_eval_900.csv`

## 결과 확인

각 stage 실행 후 생성되는 CSV 파일에서 `exact_match`, `f1` 점수를 비교합니다.

```
# 모든 결과 한 번에 확인
cat *_triviaqa_eval_900.csv
```

## 로그 확인

실행 중 상세 로그는 `./logs/` 폴더에 저장됩니다.


## 주의사항

- GPU 메모리가 부족하면 `CUDA_VISIBLE_DEVICES` 환경 변수로 GPU 조정
- `base_dir` 경로가 틀리면 모델 로드 실패하므로 반드시 확인
- bash 실행 전 코드가 실행 코드가 있는 위치로 터미널 위치를 이동시키는 것 확인

