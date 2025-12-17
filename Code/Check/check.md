# TriviaQA 평가 실행 가이드

이 문서는 conda 환경을 활성화한 뒤, bash 스크립트를 이용해 TriviaQA 평가 코드를 실행하는 방법을 설명합니다.  
모든 경로(`/home/...`, `/acpl-ssd20/...`)는 사용자의 실제 폴더 구조에 맞게 수정해야 합니다.

## 1. Conda 환경 활성화

먼저 `environment.yml` 로 만든 conda 환경을 활성화합니다.

```
# 예시: 이미 생성된 환경 이름이 growth_env 라고 가정
conda activate growth_env
```

환경 이름은 사용자가 실제로 만든 이름으로 바꿔서 사용합니다.

## 2. Python 스크립트 위치

Python 평가 스크립트(예: `j_eval_newstage1_fixed_TriviaQA.py`)는 아래와 같이 저장되어 있다고 가정합니다.

```
/home/devewha/DEMO/j_eval_newstage1_fixed_TriviaQA.py
```

이 경로 역시 사용자의 실제 파일 위치에 맞게 수정해야 합니다.

## 3. Bash 스크립트(run_triviaqa_gpu1.sh) 예시

아래는 GPU 1번을 사용해 평가를 실행하는 bash 스크립트 예시입니다.

```
#!/bin/bash
# 파일명 예: run_triviaqa_gpu1.sh
# 실행 권한 부여: chmod +x run_triviaqa_gpu1.sh
# 실행: ./run_triviaqa_gpu1.sh

export CUDA_VISIBLE_DEVICES=1        # 사용할 GPU 지정 (0,1,2,...) 중 하나

# 실행 (Python 파일 경로는 실제 위치로 수정)
python "/home/devewha/DEMO/j_eval_newstage1_fixed_TriviaQA.py" \
  2>&1 | tee "./logs/new_stage1${CUDA_VISIBLE_DEVICES}_$(date +%Y%m%d_%H%M%S).log"
```

- `CUDA_VISIBLE_DEVICES=1` 부분은 사용할 GPU 번호에 맞게 변경합니다.  
- `"/home/devewha/DEMO/j_eval_newstage1_fixed_TriviaQA.py"` 도 사용자의 실제 경로로 수정합니다.  
- `./logs` 폴더가 없다면 미리 생성합니다.

```
mkdir -p logs
```

## 4. Python 코드 내 경로 수정 안내

Python 코드(`j_eval_newstage1_fixed_TriviaQA.py`) 안에도 사용자 환경에 맞게 수정해야 할 경로들이 있습니다. 예를 들어:

```
@dataclass
class Config:
    base_dir: str = "/acpl-ssd20/25_pruning_AB_lora"  # ★ 여기를 사용자 경로로 변경
    device: str = "cuda:0"
```

- `base_dir` 는 모델 및 어댑터가 실제로 저장된 디렉터리 경로로 변경해야 합니다.  
- 예시:

```
@dataclass
class Config:
    base_dir: str = "/home/devewha/models/25_pruning_AB_lora"
    device: str = "cuda:0"
```

또한, adapter 관련 경로가 다음과 같이 정의되어 있으므로, `base_dir` 를 바르게만 잡으면 나머지는 자동으로 이어집니다.

```
@property
def a_dir(self) -> str:
    return f"{self.base_dir}/A"

@property
def adapter_dir(self) -> str:
    return f"{self.base_dir}/adapters"
```

따라서 실제 디렉터리 구조에 맞게 `base_dir` 를 한 번만 제대로 설정해 두는 것이 중요합니다.

## 5. 실행 순서 정리

1. 터미널에서 conda 환경 활성화  
   ```
   conda activate <환경이름>
   ```
2. (필요하면) `logs` 폴더 생성  
   ```
   mkdir -p /home/devewha/DEMO/logs    # 위치는 스크립트와 맞게 조정
   ```
3. bash 스크립트에 Python 파일 경로, 로그 경로, GPU 번호를 실제 환경에 맞게 수정  
4. 실행 권한 부여 후 스크립트 실행  
   ```
   chmod +x run_triviaqa_gpu1.sh
   ./run_triviaqa_gpu1.sh
   ```

> 요약:  
> - conda 환경은 사용자가 `environment.yml` 로 만든 환경 이름으로 직접 `conda activate` 해주어야 합니다.  
> - bash 스크립트와 Python 코드 안의 모든 절대 경로(`/home/...`, `/acpl-ssd20/...`)는 각자 머신에서 실제 파일이 있는 위치로 수정해야 합니다.
