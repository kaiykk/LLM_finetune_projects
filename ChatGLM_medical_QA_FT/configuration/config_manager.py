"""
Configuration Manager for Llama-Factory Training Pipeline

This module manages and validates Llama-Factory configuration files, handling
environment variable resolution, path validation, and configuration schema checking.

Usage:
    from config_manager import ConfigManager
    
    manager = ConfigManager()
    config = manager.load_config("configs/glm4_lora_sft.yaml")
    errors = manager.validate_config(config)
    if errors:
        print(f"Configuration errors: {errors}")
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass


class ConfigManager:
    """
    Manages and validates Llama-Factory configuration files.
    
    Handles:
    - YAML configuration loading
    - Environment variable resolution in paths
    - Configuration schema validation
    - Path existence and writability checks
    - Dry-run mode for testing configurations
    """
    
    # Required fields for different configuration types
    REQUIRED_FIELDS = {
        "training": ["model_name_or_path", "stage", "dataset", "template", "output_dir"],
        "merge": ["model_name_or_path", "adapter_name_or_path", "template", "export_dir"],
        "inference": ["model_name_or_path", "template", "infer_backend"]
    }
    
    # Path fields that should be validated
    PATH_FIELDS = [
        "model_name_or_path",
        "adapter_name_or_path",
        "output_dir",
        "export_dir",
        "data_dir"
    ]
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            project_root: Root directory for resolving relative paths.
                         Defaults to current working directory.
        """
        self.project_root = project_root or Path.cwd()
        logger.info(f"ConfigManager initialized with project root: {self.project_root}")
    
    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """
        Load and parse a YAML configuration file.
        
        Args:
            config_path: Path to YAML config file.
        
        Returns:
            Parsed configuration dictionary.
        
        Raises:
            ConfigurationError: If file doesn't exist or YAML is invalid.
        """
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            if config is None:
                raise ConfigurationError(f"Empty configuration file: {config_path}")
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
        
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in {config_path}: {e}"
            ) from e
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration from {config_path}: {e}"
            ) from e
    
    def resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve environment variables in configuration paths.
        
        Supports ${VAR_NAME} and $VAR_NAME syntax in path strings.
        
        Args:
            config: Configuration dictionary.
        
        Returns:
            Configuration with resolved paths.
        """
        resolved_config = config.copy()
        
        for key, value in config.items():
            if isinstance(value, str) and key in self.PATH_FIELDS:
                # Resolve environment variables
                resolved_value = os.path.expandvars(value)
                
                if resolved_value != value:
                    logger.debug(f"Resolved {key}: {value} -> {resolved_value}")
                
                resolved_config[key] = resolved_value
        
        return resolved_config
    
    def detect_config_type(self, config: Dict[str, Any]) -> str:
        """
        Detect configuration type based on fields present.
        
        Args:
            config: Configuration dictionary.
        
        Returns:
            Configuration type: "training", "merge", or "inference".
        """
        if "adapter_name_or_path" in config and "export_dir" in config:
            return "merge"
        elif "infer_backend" in config:
            return "inference"
        else:
            return "training"
    
    def validate_schema(
        self,
        config: Dict[str, Any],
        config_type: str
    ) -> List[str]:
        """
        Validate configuration against expected schema.
        
        Args:
            config: Configuration dictionary.
            config_type: Type of configuration ("training", "merge", "inference").
        
        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []
        
        # Check required fields
        required_fields = self.REQUIRED_FIELDS.get(config_type, [])
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Validate specific field types and values
        if "quantization_bit" in config:
            if config["quantization_bit"] not in [2, 3, 4, 5, 6, 8]:
                errors.append(
                    f"Invalid quantization_bit: {config['quantization_bit']}. "
                    f"Must be one of: 2, 3, 4, 5, 6, 8"
                )
        
        if "stage" in config and config["stage"] not in ["sft", "pt", "rm", "ppo", "dpo"]:
            errors.append(
                f"Invalid stage: {config['stage']}. "
                f"Must be one of: sft, pt, rm, ppo, dpo"
            )
        
        if "finetuning_type" in config and config["finetuning_type"] not in ["lora", "full", "freeze"]:
            errors.append(
                f"Invalid finetuning_type: {config['finetuning_type']}. "
                f"Must be one of: lora, full, freeze"
            )
        
        return errors
    
    def validate_paths(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate that paths in configuration exist or can be created.
        
        Args:
            config: Configuration dictionary with resolved paths.
        
        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []
        
        # Check model path exists (for training and inference)
        if "model_name_or_path" in config:
            model_path = config["model_name_or_path"]
            # Skip validation for HuggingFace model IDs (contain /)
            if "/" in model_path and not model_path.startswith("/"):
                logger.debug(f"Skipping path validation for HuggingFace model: {model_path}")
            else:
                model_path_obj = Path(model_path)
                if not model_path_obj.exists():
                    errors.append(f"Model path does not exist: {model_path}")
        
        # Check adapter path exists (for merging)
        if "adapter_name_or_path" in config:
            adapter_path = Path(config["adapter_name_or_path"])
            if not adapter_path.exists():
                errors.append(f"Adapter path does not exist: {adapter_path}")
        
        # Check output directory is writable or can be created
        for dir_field in ["output_dir", "export_dir"]:
            if dir_field in config:
                output_dir = Path(config[dir_field])
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    # Test writability
                    test_file = output_dir / ".write_test"
                    test_file.touch()
                    test_file.unlink()
                    logger.debug(f"{dir_field} is writable: {output_dir}")
                except Exception as e:
                    errors.append(
                        f"{dir_field} is not writable: {output_dir}. Error: {e}"
                    )
        
        return errors
    
    def validate_config(
        self,
        config: Dict[str, Any],
        check_paths: bool = True,
        config_type: Optional[str] = None
    ) -> List[str]:
        """
        Validate configuration against schema and optionally check paths.
        
        Args:
            config: Configuration dictionary.
            check_paths: Whether to verify paths exist. Defaults to True.
            config_type: Configuration type. Auto-detected if None.
        
        Returns:
            List of validation error messages (empty if valid).
        """
        all_errors = []
        
        # Detect config type if not provided
        if config_type is None:
            config_type = self.detect_config_type(config)
        
        logger.info(f"Validating {config_type} configuration")
        
        # Validate schema
        schema_errors = self.validate_schema(config, config_type)
        all_errors.extend(schema_errors)
        
        # Validate paths if requested
        if check_paths:
            path_errors = self.validate_paths(config)
            all_errors.extend(path_errors)
        
        if all_errors:
            logger.warning(f"Configuration validation found {len(all_errors)} errors")
        else:
            logger.info("Configuration validation passed")
        
        return all_errors
    
    def load_and_validate(
        self,
        config_path: Path,
        check_paths: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Load and validate a configuration file in one step.
        
        Args:
            config_path: Path to YAML config file.
            check_paths: Whether to verify paths exist.
            dry_run: If True, only validate without raising errors.
        
        Returns:
            Validated and resolved configuration dictionary.
        
        Raises:
            ConfigurationError: If validation fails and not in dry_run mode.
        """
        # Load configuration
        config = self.load_config(config_path)
        
        # Resolve environment variables
        config = self.resolve_paths(config)
        
        # Validate
        errors = self.validate_config(config, check_paths=check_paths)
        
        if errors:
            error_message = f"Configuration validation failed with {len(errors)} errors:\n"
            for i, error in enumerate(errors, 1):
                error_message += f"  {i}. {error}\n"
            
            if dry_run:
                logger.warning(error_message)
            else:
                raise ConfigurationError(error_message)
        
        return config


def main():
    """Command-line interface for configuration validation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate Llama-Factory configuration files"
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--no-check-paths",
        action="store_true",
        help="Skip path existence validation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without raising errors (report only)"
    )
    
    args = parser.parse_args()
    
    try:
        manager = ConfigManager()
        config = manager.load_and_validate(
            Path(args.config_file),
            check_paths=not args.no_check_paths,
            dry_run=args.dry_run
        )
        
        print(f"✓ Configuration is valid")
        print(f"  Config type: {manager.detect_config_type(config)}")
        print(f"  Model: {config.get('model_name_or_path', 'N/A')}")
        print(f"  Dataset: {config.get('dataset', 'N/A')}")
        
        return 0
    
    except ConfigurationError as e:
        print(f"✗ Configuration validation failed:")
        print(f"  {e}")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
