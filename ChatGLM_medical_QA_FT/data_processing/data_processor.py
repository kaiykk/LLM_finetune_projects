"""
Data Processor Module for Huatuo Medical QA Dataset

This module converts the Huatuo encyclopedia QA dataset from its native JSONL format
to Alpaca format required by Llama-Factory for fine-tuning ChatGLM models.

Usage:
    python -m code.data_processor \
        --input-dir /path/to/huatuo-encyclopedia-qa \
        --output-dir /path/to/huatuo-encyclopedia-qa-alpaca \
        --punctuation english
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_INPUT_DIR = "/root/autodl-tmp/huatuo-encyclopedia-qa"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/huatuo-encyclopedia-qa-alpaca"
DATASET_FILES = ["train_datasets.jsonl", "test_datasets.jsonl", "validation_datasets.jsonl"]
PUNCTUATION_STYLES = {
    "english": ", ",
    "chinese": "，"
}


class DataProcessingError(Exception):
    """Exception raised for errors during data processing."""
    pass


class ValidationError(Exception):
    """Exception raised for data validation errors."""
    pass


class DataProcessor:
    """
    Converts Huatuo medical QA dataset to Alpaca format for fine-tuning.
    
    The processor handles:
    - Format conversion from Huatuo JSONL to Alpaca JSON
    - Punctuation consistency (English or Chinese)
    - Data validation
    - Error handling for malformed records
    """
    
    def __init__(self, punctuation_style: str = "english"):
        """
        Initialize data processor.
        
        Args:
            punctuation_style: Either "english" or "chinese" for consistent punctuation.
                             Defaults to "english".
        
        Raises:
            ValueError: If punctuation_style is not valid.
        """
        if punctuation_style not in PUNCTUATION_STYLES:
            raise ValueError(
                f"Invalid punctuation style: {punctuation_style}. "
                f"Must be one of: {list(PUNCTUATION_STYLES.keys())}"
            )
        self.punctuation_style = punctuation_style
        self.separator = PUNCTUATION_STYLES[punctuation_style]
        logger.info(f"DataProcessor initialized with {punctuation_style} punctuation")
    
    def validate_record(self, record: Dict, line_num: int) -> None:
        """
        Validate a single Huatuo record structure.
        
        Args:
            record: Dictionary containing questions and answers.
            line_num: Line number in the source file (for error reporting).
        
        Raises:
            ValidationError: If required fields are missing or malformed.
        """
        missing_fields = []
        
        if "questions" not in record:
            missing_fields.append("questions")
        elif not isinstance(record["questions"], list) or len(record["questions"]) == 0:
            raise ValidationError(
                f"Line {line_num}: 'questions' must be a non-empty list"
            )
        elif not isinstance(record["questions"][0], list):
            raise ValidationError(
                f"Line {line_num}: 'questions[0]' must be a list of strings"
            )
        
        if "answers" not in record:
            missing_fields.append("answers")
        elif not isinstance(record["answers"], list):
            raise ValidationError(
                f"Line {line_num}: 'answers' must be a list"
            )
        
        if missing_fields:
            raise ValidationError(
                f"Line {line_num}: Missing required fields: {', '.join(missing_fields)}"
            )
    
    def format_record(self, record: Dict) -> Dict[str, str]:
        """
        Convert a single Huatuo record to Alpaca format.
        
        Args:
            record: Dictionary with 'questions' and 'answers' fields.
        
        Returns:
            Dictionary with 'instruction', 'input', and 'output' fields.
        """
        # Join questions with consistent punctuation
        instruction = self.separator.join(record["questions"][0])
        
        # Join answers with consistent punctuation
        output = self.separator.join(record["answers"])
        
        return {
            "instruction": instruction,
            "input": "",
            "output": output
        }
    
    def process_file(
        self,
        input_path: Path,
        output_path: Path,
        validate: bool = True
    ) -> int:
        """
        Process a single JSONL file from Huatuo to Alpaca format.
        
        Args:
            input_path: Path to input JSONL file.
            output_path: Path to output JSON file.
            validate: Whether to validate data structure. Defaults to True.
        
        Returns:
            Number of records successfully processed.
        
        Raises:
            FileNotFoundError: If input file doesn't exist.
            DataProcessingError: If processing fails.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        logger.info(f"Processing {input_path} -> {output_path}")
        
        data_list = []
        processed_count = 0
        skipped_count = 0
        
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    try:
                        # Parse JSON line
                        record = json.loads(line.strip())
                        
                        # Validate if requested
                        if validate:
                            self.validate_record(record, line_num)
                        
                        # Convert to Alpaca format
                        alpaca_record = self.format_record(record)
                        data_list.append(alpaca_record)
                        processed_count += 1
                        
                        # Log progress every 1000 records
                        if processed_count % 1000 == 0:
                            logger.debug(f"Processed {processed_count} records")
                    
                    except json.JSONDecodeError as e:
                        skipped_count += 1
                        logger.warning(
                            f"Skipping malformed JSON at line {line_num}: {e}. "
                            f"Content: {line[:100]}..."
                        )
                    except ValidationError as e:
                        skipped_count += 1
                        logger.warning(f"Skipping invalid record: {e}")
        
        except Exception as e:
            raise DataProcessingError(
                f"Failed to process {input_path}: {e}"
            ) from e
        
        # Write output file with proper resource management
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as writer:
                json.dump(data_list, writer, ensure_ascii=False, indent=4)
        except Exception as e:
            raise DataProcessingError(
                f"Failed to write output file {output_path}: {e}"
            ) from e
        
        logger.info(
            f"Conversion complete: {processed_count} records processed, "
            f"{skipped_count} records skipped"
        )
        
        return processed_count
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        validate: bool = True
    ) -> Dict[str, int]:
        """
        Process all dataset files in a directory.
        
        Args:
            input_dir: Directory containing Huatuo JSONL files.
            output_dir: Directory for Alpaca JSON output files.
            validate: Whether to validate data structure.
        
        Returns:
            Dictionary mapping filename to number of records processed.
        
        Raises:
            FileNotFoundError: If input directory doesn't exist.
            DataProcessingError: If processing fails.
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        logger.info(f"Processing directory: {input_dir} -> {output_dir}")
        
        results = {}
        for filename in DATASET_FILES:
            input_file = input_dir / filename
            output_file = output_dir / filename.replace(".jsonl", ".json")
            
            if input_file.exists():
                count = self.process_file(input_file, output_file, validate)
                results[filename] = count
            else:
                logger.warning(f"File not found, skipping: {input_file}")
        
        return results


def main():
    """Command-line interface for data processing."""
    parser = argparse.ArgumentParser(
        description="Convert Huatuo medical QA dataset to Alpaca format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory containing Huatuo JSONL files (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for Alpaca JSON files (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--punctuation",
        type=str,
        choices=["english", "chinese"],
        default="english",
        help="Punctuation style for joining questions and answers (default: english)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip data validation (faster but less safe)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Convert paths to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    try:
        # Initialize processor
        processor = DataProcessor(punctuation_style=args.punctuation)
        
        # Process all files
        results = processor.process_directory(
            input_dir,
            output_dir,
            validate=not args.no_validate
        )
        
        # Print summary
        total_records = sum(results.values())
        logger.info(f"Processing complete! Total records: {total_records}")
        for filename, count in results.items():
            logger.info(f"  {filename}: {count} records")
        
        return 0
    
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
