@echo off
REM Complete workflow script for ChatGLM medical model fine-tuning (Windows)
REM This script executes all steps from data preparation to model deployment

echo ==========================================
echo ChatGLM Medical Model Training Workflow
echo ==========================================
echo.

REM Configuration
if not defined DATA_INPUT_DIR set DATA_INPUT_DIR=/root/autodl-tmp/huatuo-encyclopedia-qa
if not defined DATA_OUTPUT_DIR set DATA_OUTPUT_DIR=/root/autodl-tmp/huatuo-encyclopedia-qa-alpaca
set CONFIG_DIR=code/llama_factory_configs

REM Step 0: Validate prerequisites
echo Step 0: Validating prerequisites...
python -m code.training_pipeline --validate-only --config-dir %CONFIG_DIR%

if errorlevel 1 (
    echo X Prerequisites validation failed. Please fix errors and try again.
    exit /b 1
)

echo √ Prerequisites validated
echo.

REM Step 1: Data conversion
echo Step 1: Converting data format (Huatuo -^> Alpaca)...
python -m code.data_processor --input-dir %DATA_INPUT_DIR% --output-dir %DATA_OUTPUT_DIR% --punctuation english --verbose

if errorlevel 1 (
    echo X Data conversion failed
    exit /b 1
)

echo √ Data conversion complete
echo.

REM Step 2: Copy data to Llama-Factory
echo Step 2: Copying converted data to Llama-Factory...
if exist "LLaMA-Factory\data" (
    copy "%DATA_OUTPUT_DIR%\train_datasets.json" "LLaMA-Factory\data\huatuo-encyclopedia-qa.json"
    echo √ Data copied to Llama-Factory
) else (
    echo Warning: LLaMA-Factory\data directory not found. Please copy data manually.
)

echo.

REM Step 3: LoRA training
echo Step 3: Starting LoRA fine-tuning...
echo This may take 2-4 hours depending on your GPU...
llamafactory-cli train %CONFIG_DIR%\glm4_lora_sft.yaml

if errorlevel 1 (
    echo X Training failed
    exit /b 1
)

echo √ Training complete
echo.

REM Step 4: Merge LoRA adapter
echo Step 4: Merging LoRA adapter with base model...
llamafactory-cli export %CONFIG_DIR%\glm4_lora_merge.yaml

if errorlevel 1 (
    echo X Model merging failed
    exit /b 1
)

echo √ Model merging complete
echo.

REM Step 5: Optional deployment
echo ==========================================
echo √ WORKFLOW COMPLETE
echo ==========================================
echo.
echo To deploy the model with vLLM, run:
echo   set API_PORT=8000
echo   llamafactory-cli api %CONFIG_DIR%\glm4_lora_vllm.yaml
echo.
echo The model will be available at: http://localhost:8000