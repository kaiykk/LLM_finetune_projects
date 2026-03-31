"""
Training Pipeline Orchestrator for ChatGLM Medical Model

This module orchestrates the complete model fine-tuning workflow from data
preparation through model deployment.

Workflow Steps:
1. Data format conversion (Huatuo -> Alpaca)
2. LoRA fine-tuning with Llama-Factory
3. Adapter merging
4. (Optional) vLLM deployment

Usage:
    python -m code.training_pipeline \
        --config-dir code/llama_factory_configs \
        --data-input-dir /path/to/huatuo-encyclopedia-qa \
        --data-output-dir /path/to/huatuo-encyclopedia-qa-alpaca
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, List
import sys

from config_manager import ConfigManager, ConfigurationError
from data_processor import DataProcessor, DataProcessingError

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Exception raised for pipeline execution errors."""
    pass


class TrainingPipeline:
    """
    Orchestrates the complete model fine-tuning workflow.
    
    Manages:
    - Data preprocessing
    - Configuration validation
    - Training execution
    - Model merging
    - Progress tracking and logging
    """
    
    def __init__(self, config_dir: Path, verbose: bool = True):
        """
        Initialize training pipeline.
        
        Args:
            config_dir: Directory containing Llama-Factory config files.
            verbose: Whether to print detailed progress information.
        """
        self.config_dir = config_dir
        self.verbose = verbose
        self.config_manager = ConfigManager()
        
        if verbose:
            logger.setLevel(logging.DEBUG)
        
        logger.info(f"TrainingPipeline initialized with config dir: {config_dir}")
    
    def validate_prerequisites(self) -> List[str]:
        """
        Validate all prerequisites before starting workflow.
        
        Checks:
        - Configuration directory exists
        - Required config files exist
        - Llama-Factory is installed
        
        Returns:
            List of validation error messages (empty if all valid).
        """
        errors = []
        
        # Check config directory
        if not self.config_dir.exists():
            errors.append(f"Configuration directory not found: {self.config_dir}")
            return errors  # Can't continue without config dir
        
        # Check required config files
        required_configs = [
            "glm4_lora_sft.yaml",
            "glm4_lora_merge.yaml"
        ]
        
        for config_file in required_configs:
            config_path = self.config_dir / config_file
            if not config_path.exists():
                errors.append(f"Required config file not found: {config_path}")
        
        # Check if llamafactory-cli is available
        try:
            result = subprocess.run(
                ["llamafactory-cli", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                errors.append("llamafactory-cli is not properly installed")
        except FileNotFoundError:
            errors.append("llamafactory-cli command not found. Please install Llama-Factory")
        except Exception as e:
            errors.append(f"Failed to check llamafactory-cli: {e}")
        
        return errors
    
    def run_step(
        self,
        step_name: str,
        command: List[str],
        cwd: Optional[Path] = None
    ) -> None:
        """
        Execute a single pipeline step with error handling.
        
        Args:
            step_name: Human-readable name of the step.
            command: Command to execute as list of strings.
            cwd: Working directory for command execution.
        
        Raises:
            PipelineError: If step execution fails.
        """
        logger.info(f"Starting step: {step_name}")
        logger.debug(f"Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if self.verbose and result.stdout:
                logger.debug(f"Output: {result.stdout}")
            
            logger.info(f"✓ Step completed: {step_name}")
        
        except subprocess.CalledProcessError as e:
            error_msg = (
                f"Step failed: {step_name}\n"
                f"Command: {' '.join(command)}\n"
                f"Exit code: {e.returncode}\n"
                f"Error output: {e.stderr}"
            )
            logger.error(error_msg)
            raise PipelineError(error_msg) from e
        
        except Exception as e:
            error_msg = f"Unexpected error in step '{step_name}': {e}"
            logger.error(error_msg)
            raise PipelineError(error_msg) from e
    
    def step_data_conversion(
        self,
        input_dir: Path,
        output_dir: Path,
        punctuation: str = "english"
    ) -> None:
        """
        Execute data format conversion step.
        
        Args:
            input_dir: Raw Huatuo dataset directory.
            output_dir: Alpaca format output directory.
            punctuation: Punctuation style ("english" or "chinese").
        
        Raises:
            PipelineError: If data conversion fails.
        """
        logger.info("=" * 60)
        logger.info("STEP 1: Data Format Conversion")
        logger.info("=" * 60)
        
        try:
            processor = DataProcessor(punctuation_style=punctuation)
            results = processor.process_directory(input_dir, output_dir, validate=True)
            
            total = sum(results.values())
            logger.info(f"✓ Data conversion complete: {total} total records")
        
        except Exception as e:
            raise PipelineError(f"Data conversion failed: {e}") from e
    
    def step_lora_training(self, config_file: str) -> None:
        """
        Execute LoRA fine-tuning step.
        
        Args:
            config_file: Name of training config file (e.g., "glm4_lora_sft.yaml").
        
        Raises:
            PipelineError: If training fails.
        """
        logger.info("=" * 60)
        logger.info("STEP 2: LoRA Fine-Tuning")
        logger.info("=" * 60)
        
        config_path = self.config_dir / config_file
        
        # Validate configuration
        try:
            config = self.config_manager.load_and_validate(
                config_path,
                check_paths=False  # Skip path checks as model may be downloaded
            )
        except ConfigurationError as e:
            raise PipelineError(f"Configuration validation failed: {e}") from e
        
        # Execute training
        command = ["llamafactory-cli", "train", str(config_path)]
        self.run_step("LoRA Training", command)
    
    def step_model_merging(self, config_file: str) -> None:
        """
        Execute model merging step.
        
        Args:
            config_file: Name of merge config file (e.g., "glm4_lora_merge.yaml").
        
        Raises:
            PipelineError: If merging fails.
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Model Merging")
        logger.info("=" * 60)
        
        config_path = self.config_dir / config_file
        
        # Validate configuration
        try:
            config = self.config_manager.load_and_validate(config_path, check_paths=True)
        except ConfigurationError as e:
            raise PipelineError(f"Configuration validation failed: {e}") from e
        
        # Execute merging
        command = ["llamafactory-cli", "export", str(config_path)]
        self.run_step("Model Merging", command)
    
    def run_complete_workflow(
        self,
        data_input_dir: Path,
        data_output_dir: Path,
        skip_data_processing: bool = False,
        skip_training: bool = False,
        skip_merging: bool = False
    ) -> None:
        """
        Execute complete workflow from data processing to model merging.
        
        Args:
            data_input_dir: Raw Huatuo dataset directory.
            data_output_dir: Alpaca format output directory.
            skip_data_processing: Skip data conversion if already done.
            skip_training: Skip training step (for testing).
            skip_merging: Skip merging step (for testing).
        
        Raises:
            PipelineError: If any step fails.
        """
        logger.info("=" * 60)
        logger.info("ChatGLM Medical Model Training Pipeline")
        logger.info("=" * 60)
        
        # Validate prerequisites
        logger.info("Validating prerequisites...")
        errors = self.validate_prerequisites()
        if errors:
            error_msg = "Prerequisites validation failed:\n"
            for i, error in enumerate(errors, 1):
                error_msg += f"  {i}. {error}\n"
            raise PipelineError(error_msg)
        
        logger.info("✓ Prerequisites validated")
        
        try:
            # Step 1: Data conversion
            if not skip_data_processing:
                self.step_data_conversion(data_input_dir, data_output_dir)
            else:
                logger.info("Skipping data conversion (already done)")
            
            # Step 2: LoRA training
            if not skip_training:
                self.step_lora_training("glm4_lora_sft.yaml")
            else:
                logger.info("Skipping training step")
            
            # Step 3: Model merging
            if not skip_merging:
                self.step_model_merging("glm4_lora_merge.yaml")
            else:
                logger.info("Skipping merging step")
            
            logger.info("=" * 60)
            logger.info("✓ PIPELINE COMPLETE")
            logger.info("=" * 60)
        
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(f"Unexpected pipeline error: {e}") from e


def main():
    """Command-line interface for training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Execute ChatGLM medical model training pipeline"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="code/llama_factory_configs",
        help="Directory containing Llama-Factory config files"
    )
    parser.add_argument(
        "--data-input-dir",
        type=str,
        default="/root/autodl-tmp/huatuo-encyclopedia-qa",
        help="Input directory with raw Huatuo dataset"
    )
    parser.add_argument(
        "--data-output-dir",
        type=str,
        default="/root/autodl-tmp/huatuo-encyclopedia-qa-alpaca",
        help="Output directory for Alpaca format dataset"
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data conversion step"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training step (for testing)"
    )
    parser.add_argument(
        "--skip-merging",
        action="store_true",
        help="Skip merging step (for testing)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate prerequisites without executing"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        pipeline = TrainingPipeline(
            config_dir=Path(args.config_dir),
            verbose=args.verbose
        )
        
        if args.validate_only:
            logger.info("Running prerequisites validation only...")
            errors = pipeline.validate_prerequisites()
            if errors:
                print("✗ Validation failed:")
                for i, error in enumerate(errors, 1):
                    print(f"  {i}. {error}")
                return 1
            else:
                print("✓ All prerequisites validated successfully")
                return 0
        
        # Run complete workflow
        pipeline.run_complete_workflow(
            data_input_dir=Path(args.data_input_dir),
            data_output_dir=Path(args.data_output_dir),
            skip_data_processing=args.skip_data,
            skip_training=args.skip_training,
            skip_merging=args.skip_merging
        )
        
        return 0
    
    except PipelineError as e:
        logger.error(f"Pipeline failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())