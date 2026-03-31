# ChatGLM Medical Model Fine-Tuning Project

This project fine-tunes the ChatGLM-4 model on medical question-answering data using the Huatuo-26M encyclopedia QA dataset. The pipeline uses Llama-Factory for training and vLLM for high-performance inference deployment.

## Project Overview

**Goal**: Create a specialized medical chatbot that can answer encyclopedia-style medical questions in Chinese.

**Approach**:
- Base Model: GLM-4-9B-Chat (9 billion parameter conversational model)
- Training Method: LoRA (Low-Rank Adaptation) with 8-bit quantization
- Dataset: Huatuo-26M Encyclopedia QA (~100K medical QA pairs)
- Deployment: vLLM inference server

**Workflow**:
```
Raw Huatuo Dataset → Data Conversion → LoRA Training → Model Merging → vLLM Deployment
```

## Project Structure

```
optimized_code/
├── data_processing/       # Data conversion modules
│   ├── data_processor.py    # Main conversion tool
│   └── huatuo_dataset.py    # HuggingFace dataset loader
├── configuration/         # Llama-Factory configs
│   ├── config_manager.py    # Config validation tool
│   ├── glm4_lora_sft.yaml  # LoRA training config
│   ├── glm4_full_sft.yaml  # Full fine-tuning config
│   ├── glm4_lora_merge.yaml # Model merging config
│   └── glm4_lora_vllm.yaml # vLLM inference config
├── training_pipeline/     # Workflow orchestration
│   └── training_pipeline.py # Complete pipeline tool
├── 04_automation/            # Automated workflow scripts
│   ├── run_workflow.sh      # Linux/Mac script
│   └── run_workflow.bat     # Windows script
└── 05_testing/               # Test suite
    ├── __init__.py
    └── test_data_processor.py
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU training)
- 24GB+ GPU memory

### Environment Setup

1. **Install Llama-Factory**:

2. **Install vLLM**:
```bash
pip install vllm==0.7.0 --extra-index-url https://download.pytorch.org/whl/cu121
```

3. **Install All Dependencies**:
```bash
pip install -r requirements.txt
```

## Quick Start

#### Step 1: Data Format Conversion

```bash
python 01_data_processing/data_processor.py \
    --input-dir /root/autodl-tmp/huatuo-encyclopedia-qa \
    --output-dir /root/autodl-tmp/huatuo-encyclopedia-qa-alpaca \
    --punctuation english \
    --verbose
```

#### Step 2: Configure Dataset in Llama-Factory

Add to `LLaMA-Factory/data/dataset_info.json`:

```json
"huatuo": {
    "file_name": "huatuo-encyclopedia-qa.json",
    "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output"
    }
}
```

Copy data file:
```bash
cp /root/autodl-tmp/huatuo-encyclopedia-qa-alpaca/train_datasets.json \
   LLaMA-Factory/data/huatuo-encyclopedia-qa.json
```

#### Step 3: LoRA Fine-Tuning

```bash
llamafactory-cli train 02_configuration/glm4_lora_sft.yaml
```

Training time: ~2-4 hours on RTX 3090

#### Step 4: Merge LoRA Adapter

```bash
llamafactory-cli export 02_configuration/glm4_lora_merge.yaml
```

#### Step 5: Deploy with vLLM

```bash
API_PORT=8000 llamafactory-cli api 02_configuration/glm4_lora_vllm.yaml
```

Server available at: `http://localhost:8000`

## Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` with your paths:
```bash
DATA_INPUT_DIR=/your/path/to/huatuo-encyclopedia-qa
DATA_OUTPUT_DIR=/your/path/to/huatuo-encyclopedia-qa-alpaca
CHECKPOINT_DIR=/your/path/to/checkpoints
```

### Configuration Files

All configs in `configuration/`:

- **glm4_lora_sft.yaml**: LoRA training (10GB GPU memory)
- **glm4_full_sft.yaml**: Full fine-tuning (120GB GPU memory)
- **glm4_lora_merge.yaml**: Merge LoRA with base model
- **glm4_lora_vllm.yaml**: vLLM inference deployment

## Dataset Information

### Huatuo-26M Encyclopedia QA

- **Source**: https://github.com/FreedomIntelligence/Huatuo-26M
- **Size**: ~100K question-answer pairs
- **Language**: Chinese
- **Domain**: Medical encyclopedia knowledge

**Data Format**:

Original (Huatuo):
```json
{
  "questions": [["什么是高血压？"]],
  "answers": ["高血压是指血压持续升高的疾病"]
}
```

Converted (Alpaca):
```json
{
  "instruction": "什么是高血压？",
  "input": "",
  "output": "高血压是指血压持续升高的疾病"
}
```

## Usage Examples

### Validate Configuration

```bash
python configuration/config_manager.py \
    configuration/glm4_lora_sft.yaml \
    --dry-run
```

### Validate Prerequisites

```bash
python training_pipeline/training_pipeline.py --validate-only
```

### Skip Completed Steps

```bash
# Skip data conversion if already done
python training_pipeline/training_pipeline.py --skip-data

# Skip training for testing
python training_pipeline/training_pipeline.py --skip-training --skip-merging
```

### Custom Punctuation Style

```bash
python data_processing/data_processor.py \
    --punctuation chinese \
    --input-dir /path/to/input \
    --output-dir /path/to/output
```

## Testing

Run unit tests:
```bash
cd 05_testing
pytest test_data_processor.py -v
```

## Citation

```bibtex
@misc{li2023huatuo26m,
      title={Huatuo-26M, a Large-scale Chinese Medical QA Dataset}, 
      author={Jianquan Li and Xidong Wang and Xiangbo Wu and Zhiyi Zhang and Xiaolong Xu and Jie Fu and Prayag Tiwari and Xiang Wan and Benyou Wang},
      year={2023},
      eprint={2305.01526},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


