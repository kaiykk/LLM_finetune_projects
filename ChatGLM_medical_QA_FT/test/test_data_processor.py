"""
Unit tests for DataProcessor module.

Tests cover:
- Punctuation consistency
- File handle management
- Error handling
- Data validation
"""

import json
import pytest
from pathlib import Path
import tempfile
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processor import DataProcessor, ValidationError, DataProcessingError


class TestDataProcessor:
    """Test suite for DataProcessor class."""
    
    def test_initialization_valid_punctuation(self):
        """Test processor initializes with valid punctuation styles."""
        processor_en = DataProcessor(punctuation_style="english")
        assert processor_en.separator == ", "
        
        processor_cn = DataProcessor(punctuation_style="chinese")
        assert processor_cn.separator == "，"
    
    def test_initialization_invalid_punctuation(self):
        """Test processor raises error for invalid punctuation style."""
        with pytest.raises(ValueError):
            DataProcessor(punctuation_style="invalid")
    
    def test_format_record_english_punctuation(self):
        """Test record formatting with English punctuation."""
        processor = DataProcessor(punctuation_style="english")
        
        record = {
            "questions": [["What is diabetes?", "Define diabetes"]],
            "answers": ["A metabolic disease", "High blood sugar condition"]
        }
        
        result = processor.format_record(record)
        
        assert result["instruction"] == "What is diabetes?, Define diabetes"
        assert result["output"] == "A metabolic disease, High blood sugar condition"
        assert result["input"] == ""
    
    def test_format_record_chinese_punctuation(self):
        """Test record formatting with Chinese punctuation."""
        processor = DataProcessor(punctuation_style="chinese")
        
        record = {
            "questions": [["什么是糖尿病？", "糖尿病的定义"]],
            "answers": ["一种代谢疾病", "血糖持续升高的状态"]
        }
        
        result = processor.format_record(record)
        
        assert result["instruction"] == "什么是糖尿病？，糖尿病的定义"
        assert result["output"] == "一种代谢疾病，血糖持续升高的状态"
    
    def test_validate_record_valid(self):
        """Test validation passes for valid records."""
        processor = DataProcessor()
        
        valid_record = {
            "questions": [["Question 1", "Question 2"]],
            "answers": ["Answer 1", "Answer 2"]
        }
        
        # Should not raise
        processor.validate_record(valid_record, line_num=1)
    
    def test_validate_record_missing_questions(self):
        """Test validation fails for missing questions field."""
        processor = DataProcessor()
        
        invalid_record = {
            "answers": ["Answer 1"]
        }
        
        with pytest.raises(ValidationError) as exc_info:
            processor.validate_record(invalid_record, line_num=5)
        
        assert "questions" in str(exc_info.value)
        assert "Line 5" in str(exc_info.value)

    def test_validate_record_missing_answers(self):
        """Test validation fails for missing answers field."""
        processor = DataProcessor()
        
        invalid_record = {
            "questions": [["Question 1"]]
        }
        
        with pytest.raises(ValidationError) as exc_info:
            processor.validate_record(invalid_record, line_num=10)
        
        assert "answers" in str(exc_info.value)
        assert "Line 10" in str(exc_info.value)
    
    def test_process_file_creates_output(self):
        """Test that process_file creates valid output file."""
        processor = DataProcessor(punctuation_style="english")
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            input_path = Path(f.name)
            # Write test data
            test_data = [
                {"questions": [["Q1", "Q2"]], "answers": ["A1", "A2"]},
                {"questions": [["Q3"]], "answers": ["A3"]}
            ]
            for record in test_data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                output_path = Path(f.name)
            
            # Process file
            count = processor.process_file(input_path, output_path, validate=True)
            
            # Verify results
            assert count == 2
            assert output_path.exists()
            
            # Check output content
            with open(output_path, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
            
            assert len(output_data) == 2
            assert output_data[0]["instruction"] == "Q1, Q2"
            assert output_data[0]["output"] == "A1, A2"
        
        finally:
            # Cleanup
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
    
    def test_process_file_handles_malformed_json(self):
        """Test that malformed JSON lines are skipped with logging."""
        processor = DataProcessor()
        
        # Create temporary input file with malformed JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            input_path = Path(f.name)
            f.write('{"questions": [["Q1"]], "answers": ["A1"]}\n')
            f.write('{"invalid json\n')  # Malformed
            f.write('{"questions": [["Q2"]], "answers": ["A2"]}\n')
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                output_path = Path(f.name)
            
            # Should process valid records and skip malformed
            count = processor.process_file(input_path, output_path, validate=True)
            
            assert count == 2  # Only 2 valid records
            
            with open(output_path, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
            
            assert len(output_data) == 2
        
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])