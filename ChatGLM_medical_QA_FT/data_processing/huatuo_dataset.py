"""
Huatuo Medical Encyclopedia QA Dataset Loader

This module provides a HuggingFace datasets loader for the Huatuo-26M medical
encyclopedia question-answering dataset.

Dataset Information:
- Name: Huatuo-26M Encyclopedia QA
- Size: ~100K QA pairs
- Language: Chinese
- Domain: Medical encyclopedia knowledge
- Format: JSONL with questions and answers arrays

Usage:
    from datasets import load_dataset
    
    # Load from local directory
    dataset = load_dataset("huatuo_dataset.py")
    
    # Access splits
    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]
    
    # Iterate over examples
    for example in train_data:
        questions = example["questions"]  # List of question phrasings
        answers = example["answers"]      # List of answer variations
"""

from datasets import (
    DatasetInfo,
    Features,
    Split,
    SplitGenerator,
    GeneratorBasedBuilder,
    Value,
    Sequence
)
import json
from pathlib import Path
from typing import Generator, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class HuatuoDataset(GeneratorBasedBuilder):
    """
    HuggingFace dataset builder for Huatuo medical encyclopedia QA.
    
    This dataset contains medical question-answer pairs from the Huatuo-26M project,
    focusing on encyclopedia-style medical knowledge.
    """
    
    VERSION = "1.0.0"
    
    def _info(self) -> DatasetInfo:
        """
        Returns dataset metadata and feature schema.
        
        Returns:
            DatasetInfo object with features, citation, and homepage.
        """
        return DatasetInfo(
            features=Features({
                "questions": Sequence(Value("string")),
                "answers": Sequence(Value("string"))
            }),
            supervised_keys=("questions", "answers"),
            homepage="https://github.com/FreedomIntelligence/Huatuo-26M",
            citation='''
            @misc{li2023huatuo26m,
                  title={Huatuo-26M, a Large-scale Chinese Medical QA Dataset}, 
                  author={Jianquan Li and Xidong Wang and Xiangbo Wu and Zhiyi Zhang and Xiaolong Xu and Jie Fu and Prayag Tiwari and Xiang Wan and Benyou Wang},
                  year={2023},
                  eprint={2305.01526},
                  archivePrefix={arXiv},
                  primaryClass={cs.CL}
            }
            ''',
            description="Huatuo-26M medical encyclopedia QA dataset for training medical chatbots"
        )
    
    def _split_generators(self, dl_manager) -> List[SplitGenerator]:
        """
        Returns split generators for train/validation/test splits.
        
        Validates that all required data files exist before proceeding.
        
        Args:
            dl_manager: Download manager (not used for local files).
        
        Returns:
            List of SplitGenerator objects for each data split.
        
        Raises:
            FileNotFoundError: If any required data file is missing.
        """
        # Define expected file paths
        train_path = "train_datasets.jsonl"
        validation_path = "validation_datasets.jsonl"
        test_path = "test_datasets.jsonl"
        
        # Validate all files exist
        missing_files = []
        for filepath in [train_path, validation_path, test_path]:
            if not Path(filepath).exists():
                missing_files.append(filepath)
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing required data files: {', '.join(missing_files)}. "
                f"Please ensure all dataset files are in the current directory."
            )
        
        logger.info("All dataset files found, creating split generators")
        
        return [
            SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"filepath": train_path}
            ),
            SplitGenerator(
                name=Split.VALIDATION,
                gen_kwargs={"filepath": validation_path}
            ),
            SplitGenerator(
                name=Split.TEST,
                gen_kwargs={"filepath": test_path}
            ),
        ]
    
    def _generate_examples(
        self,
        filepath: str
    ) -> Generator[Tuple[int, Dict], None, None]:
        """
        Yields examples from the dataset file.
        
        Args:
            filepath: Path to JSONL data file.
        
        Yields:
            Tuple of (example_id, example_dict) where example_dict contains
            'questions' and 'answers' fields.
        
        Raises:
            FileNotFoundError: If filepath doesn't exist.
            json.JSONDecodeError: If a line contains malformed JSON.
        """
        filepath_obj = Path(filepath)
        
        if not filepath_obj.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Loading examples from {filepath}")
        
        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                try:
                    # Parse JSON line
                    data = json.loads(row)
                    
                    # Yield example
                    yield id_, {
                        "questions": data["questions"],
                        "answers": data["answers"],
                    }
                
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Malformed JSON at line {id_ + 1} in {filepath}: {e}"
                    )
                    raise
                except KeyError as e:
                    logger.error(
                        f"Missing required field {e} at line {id_ + 1} in {filepath}"
                    )
                    raise


if __name__ == '__main__':
    # Example usage
    from datasets import load_dataset
    
    print("Loading Huatuo medical encyclopedia QA dataset...")
    dataset = load_dataset("huatuo_dataset.py")
    
    print(f"Dataset loaded successfully!")
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}")
    print(f"Test examples: {len(dataset['test'])}")
    
    # Show first example
    print("\nFirst training example:")
    example = dataset['train'][0]
    print(f"Questions: {example['questions']}")
    print(f"Answers: {example['answers']}")
