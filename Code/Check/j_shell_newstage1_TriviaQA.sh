#!/bin/bash
# 파일명 예: run_triviaqa_gpu1.sh
# 실행 권한 부여: chmod +x run_triviaqa_gpu1.sh
# 실행: ./run_triviaqa_gpu1.sh

export CUDA_VISIBLE_DEVICES=1        # 사용할 GPU 지정 (0,1,2,... 중 하나)

# 실행
python "/home/devewha/Growth/Growth/Code/Check/j_eval_newstage1_fixed_TriviaQA.py" \
  2>&1 | tee "./logs/new_stage1${CUDA_VISIBLE_DEVICES}_$(date +%Y%m%d_%H%M%S).log"
