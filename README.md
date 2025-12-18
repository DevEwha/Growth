# ProgressiveServe: Progressive Model Loading and Recovery for Mitigating Cold Start in Serverless LLM Serving

**서버리스 LLM 콜드 스타트 완화를 위한 점진적 모델 로딩 및 복구 기법**

## 📖 Overview

ProgressiveServe는 서버리스 환경에서 LLM(Large Language Model) 서빙 시 발생하는 콜드 스타트 문제를 해결하기 위한 점진적 모델 로딩 및 복구 파이프라인입니다. 프루닝된 경량 모델을 우선 서빙하고 백그라운드에서 전체 모델을 점진적으로 복구하여, TTFT(Time-To-First-Token)를 21.1% 단축하면서도 최종 정확도를 원본 모델 수준으로 유지합니다.

### Key Features

- **각도 기반 연속 레이어 프루닝**: 코사인 유사도를 활용한 레이어 중요도 평가 및 선택적 제거
- **3단계 점진적 복구**: 경량 모델(단계 1) → 중간 모델(단계 2) → 전체 모델(단계 3) 순차 로딩
- **LoRA 어댑터 기반 성능 복구**: 각 단계별 특화된 어댑터를 통한 성능 저하 완화
- **PassLayer 메커니즘**: 서비스 중단 없는 동적 레이어 교체 구조

### Performance

| Method | TTFT (s) | EM (%) | F1 (%) |
|--------|----------|--------|--------|
| ServerlessLLM | 114 | 55.67 | 66.11 |
| **ProgressiveServe (Stage 1)** | **90** | 48.22 | 54.26 |
| **ProgressiveServe (Final)** | **90** | **55.67** | **66.11** |

*Evaluation on TriviaQA validation set with Llama2-7B model*

***

## 🗂️ Project Structure

```
Growth/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── configs/                           # Configuration files
│   ├── model_config.yaml             # Model architecture settings
│   └── pruning_config.yaml           # Pruning parameters
├── src/                              # Source code
│   ├── pruning/                      # Layer pruning implementation
│   │   ├── angle_based_pruning.py   # Cosine similarity-based pruning
│   │   └── layer_selection.py       # Layer importance evaluation
│   ├── lora/                         # LoRA adapter training and management
│   │   ├── adapter_trainer.py       # LoRA fine-tuning
│   │   └── adapter_manager.py       # Multi-stage adapter switching
│   ├── progressive_loading/          # Progressive loading pipeline
│   │   ├── stage_manager.py         # 3-stage loading orchestration
│   │   ├── pass_layer.py            # PassLayer placeholder implementation
│   │   └── model_merger.py          # Layer group integration
│   ├── serving/                      # Serverless serving infrastructure
│   │   ├── rayserve_handler.py      # Ray Serve deployment
│   │   └── inference_server.py      # Inference endpoint
│   └── utils/                        # Utility functions
│       ├── nfs_loader.py            # NFS-based model loading
│       └── metrics.py               # TTFT, EM, F1 calculators
├── scripts/                          # Automation scripts
│   ├── prepare_pruned_models.sh     # Offline pruning pipeline
│   ├── train_lora_adapters.sh       # LoRA adapter training
│   └── deploy_serve.sh              # Serverless deployment
├── experiments/                      # Experimental results
│   ├── data/                        # Evaluation datasets
│   │   ├── triviaqa_samples.json   # TriviaQA validation subset
│   │   └── squad_train.json        # SQuAD training data for LoRA
│   ├── results/                     # Experiment outputs
│   │   ├── cold_start_metrics.csv  # TTFT measurements
│   │   ├── qa_performance.csv      # EM/F1 scores per stage
│   │   └── ablation_study.csv      # Component-wise analysis
│   └── notebooks/                   # Analysis notebooks
│       ├── visualize_results.ipynb # Performance visualization
│       └── layer_importance.ipynb  # Pruning strategy analysis
├── models/                           # Pre-trained model artifacts
│   ├── llama2-7b-pruned/            # Pruned layer groups
│   │   ├── group_A/                 # Layers 1-20, 29-32
│   │   ├── group_B/                 # Layers 21-24
│   │   └── group_C/                 # Layers 25-28
│   └── lora_adapters/               # Trained LoRA adapters
│       ├── adapter_A/               # Stage 1 adapter
│       └── adapter_AB/              # Stage 2 adapter
└── tests/                            # Unit and integration tests
    ├── test_pruning.py
    ├── test_lora.py
    └── test_progressive_loading.py
```

***

## 🚀 Quick Start

### Prerequisites

- Ubuntu 24.04 or compatible Linux distribution
- NVIDIA GPU with CUDA 13.0+ support (tested on RTX 5090)
- NVIDIA Driver 580.65.06+
- Python 3.9+
- 16Gbps+ network for NFS storage access

### Installation

```bash
# Clone repository
git clone https://github.com/DevEwha/Growth.git
cd Growth

# Install dependencies
pip install -r requirements.txt

# Install Ray Serve for serverless deployment
pip install ray[serve]==2.9.0

# Download base model (Llama2-7B)
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir models/llama2-7b-base
```

***

## 📦 Build and Prepare Models

### Step 1: Generate Pruned Model Groups

```bash
# Run angle-based layer pruning (offline stage)
bash scripts/prepare_pruned_models.sh \
    --model_path models/llama2-7b-base \
    --output_dir models/llama2-7b-pruned \
    --prune_layers 21-28
```

**Output**: Creates layer groups A (1-20, 29-32), B (21-24), C (25-28) in FP16 Safetensors format.

### Step 2: Train LoRA Adapters

```bash
# Train stage-specific LoRA adapters
bash scripts/train_lora_adapters.sh \
    --base_model models/llama2-7b-base \
    --pruned_groups models/llama2-7b-pruned \
    --dataset experiments/data/squad_train.json \
    --output_dir models/lora_adapters
```

**Output**: Generates `adapter_A` (optimized for Stage 1) and `adapter_AB` (optimized for Stage 2).

***

## 🔧 How to Run

### Local Testing

```bash
# Test progressive loading pipeline
python src/serving/inference_server.py \
    --config configs/model_config.yaml \
    --mode progressive \
    --input "What is the capital of France?"
```

### Serverless Deployment with Ray Serve

```bash
# Deploy to Ray Serve (simulates serverless cold start)
bash scripts/deploy_serve.sh \
    --nfs_mount /mnt/nfs_models \
    --gpu_count 1 \
    --port 8000

# Send inference request
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Explain serverless computing", "max_tokens": 50}'
```

**Cold Start Measurement**: Each request creates a fresh Ray Actor to measure TTFT from scratch.

***

## 🧪 Experiments and Evaluation

### Reproduce Paper Results

```bash
# Run full evaluation pipeline (TriviaQA zero-shot)
python experiments/evaluate_cold_start.py \
    --dataset experiments/data/triviaqa_samples.json \
    --baseline serverlessllm \
    --proposed progressiveserve \
    --trials 10 \
    --clear_cache

# Results will be saved to experiments/results/
```

### Evaluation Metrics

- **TTFT (Time-To-First-Token)**: Wall-clock time from actor creation to first token generation
- **EM (Exact Match)**: Percentage of predictions exactly matching ground truth
- **F1 Score**: Token-level overlap between prediction and ground truth

### Available Experimental Data

| File | Description |
|------|-------------|
| `cold_start_metrics.csv` | TTFT measurements across 10 trials with cache cleared |
| `qa_performance.csv` | EM/F1 scores for each progressive stage (1→2→3) |
| `ablation_study.csv` | Performance comparison: w/ vs w/o LoRA adapters |

### Visualization

```bash
# Generate performance plots
jupyter notebook experiments/notebooks/visualize_results.ipynb
```

***

## 📊 Sample Data

### TriviaQA Validation Samples

Located in `experiments/data/triviaqa_samples.json`, contains 100 question-answer pairs for zero-shot evaluation:

```json
{
  "question": "Who was the first president of the United States?",
  "answer": "George Washington",
  "context": "..."
}
```

### SQuAD Training Data

Located in `experiments/data/squad_train.json`, used for LoRA adapter fine-tuning:

```json
{
  "context": "...",
  "question": "...",
  "answers": [{"text": "...", "answer_start": 0}]
}
```

***

## 🛠️ Configuration

### Model Configuration (`configs/model_config.yaml`)

```yaml
model:
  name: llama2-7b
  precision: fp16
  nfs_path: /mnt/nfs_models/llama2-7b-pruned

progressive_loading:
  stages:
    - name: stage1
      layers: [1-20, 29-32]
      lora_adapter: adapter_A
    - name: stage2
      layers: [1-24, 29-32]
      lora_adapter: adapter_AB
    - name: stage3
      layers: [1-32]
      lora_adapter: null

serving:
  framework: rayserve
  gpu_memory: 24GB
  max_concurrent_requests: 1
```

***

## 🔬 Used Open Source Libraries

| Library | Version | Purpose | License |
|---------|---------|---------|---------|
| PyTorch | 2.0+ | Deep learning framework | BSD-3-Clause |
| Transformers | 4.35+ | Hugging Face model hub | Apache 2.0 |
| PEFT | 0.7+ | LoRA adapter implementation | Apache 2.0 |
| Ray Serve | 2.9+ | Serverless deployment framework | Apache 2.0 |
| Safetensors | 0.4+ | Efficient model serialization | Apache 2.0 |
| Datasets | 2.14+ | SQuAD/TriviaQA data loading | Apache 2.0 |

### Key References

1. **ServerlessLLM**: Y. Fu et al., "ServerlessLLM: Low-latency serverless inference for large language models," USENIX OSDI 2024
2. **LoRA**: E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," arXiv:2106.09685
3. **Layer Pruning**: A. Gromov et al., "The unreasonable ineffectiveness of the deeper layers," arXiv:2403.17887
4. **Llama 2**: H. Touvron et al., "Llama 2: Open Foundation and Fine-Tuned Chat Models," arXiv:2307.09288

***

## 🧑‍💻 Authors

**Equal Contribution**

- **Nadam Park** (parknd@ewhain.net)
- **Nakyeong Lee** (rinarina0429@ewha.ac.kr)
- **Juwon Lee** (juwonlee.cse@gmail.com)

**Advisor**

- **Jaehyeong Sim** (jh.sim@ewha.ac.kr)

Department of Computer Science and Engineering, Ewha Womans University

***

## 📄 Citation

If you use this code in your research, please cite our paper

***

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

***
