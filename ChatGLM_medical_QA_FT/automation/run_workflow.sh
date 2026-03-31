#!/bin/bash
# Complete workflow script for ChatGLM medical model fine-tuning
# This script executes all steps from data preparation to model deployment

set -e  # Exit on any error

echo "=========================================="
echo "ChatGLM Medical Model Training Workflow"
echo "=========================================="
echo ""

# Configuration
DATA_INPUT_DIR="${DATA_INPUT_DIR:-/root/autodl-tmp/huatuo-encyclopedia-qa}"
DATA_OUTPUT_DIR="${DATA_OUTPUT_DIR:-/root/autodl-tmp/huatuo-encyclopedia-qa-alpaca}"
CONFIG_DIR="code/llama_factory_configs"

# Step 0: Validate prerequisites
echo "Step 0: Validating prerequisites..."
python -m code.training_pipeline --validate-only --config-dir "$CONFIG_DIR"

if [ $? -ne 0 ]; then
    echo "✗ Prerequisites validation failed. Please fix errors and try again."
    exit 1
fi

echo "✓ Prerequisites validated"
echo ""

# Step 1: Data conversion
echo "Step 1: Converting data format (Huatuo -> Alpaca)..."
python -m code.data_processor \
    --input-dir "$DATA_INPUT_DIR" \
    --output-dir "$DATA_OUTPUT_DIR" \
    --punctuation english \
    --verbose

if [ $? -ne 0 ]; then
    echo "✗ Data conversion failed"
    exit 1
fi

echo "✓ Data conversion complete"
echo ""

# Step 2: Copy data to Llama-Factory
echo "Step 2: Copying converted data to Llama-Factory..."
if [ -d "LLaMA-Factory/data" ]; then
    cp "$DATA_OUTPUT_DIR/train_datasets.json" LLaMA-Factory/data/huatuo-encyclopedia-qa.json
    echo "✓ Data copied to Llama-Factory"
else
    echo "⚠ Warning: LLaMA-Factory/data directory not found. Please copy data manually."
fi

echo ""

# Step 3: LoRA training
echo "Step 3: Starting LoRA fine-tuning..."
echo "This may take 2-4 hours depending on your GPU..."
llamafactory-cli train "$CONFIG_DIR/glm4_lora_sft.yaml"

if [ $? -ne 0 ]; then
    echo "✗ Training failed"
    exit 1
fi

echo "✓ Training complete"
echo ""

# Step 4: Merge LoRA adapter
echo "Step 4: Merging LoRA adapter with base model..."
llamafactory-cli export "$CONFIG_DIR/glm4_lora_merge.yaml"

if [ $? -ne 0 ]; then
    echo "✗ Model merging failed"
    exit 1
fi

echo "✓ Model merging complete"
echo ""

# Step 5: Optional deployment
echo "=========================================="
echo "✓ WORKFLOW COMPLETE"
echo "=========================================="
echo ""
echo "To deploy the model with vLLM, run:"
echo "  API_PORT=8000 llamafactory-cli api $CONFIG_DIR/glm4_lora_vllm.yaml"
echo ""
echo "The model will be available at: http://localhost:8000"