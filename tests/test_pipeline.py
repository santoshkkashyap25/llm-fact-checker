import pytest
from pipeline import run_fact_checking_pipeline
from core.vector_db import vector_db
from unittest.mock import Mock, patch
import pandas as pd

class TestPipeline:
    """Test suite for fact-checking pipeline"""
    
    @pytest.fixture(autouse=True)
    def setup_db(self, tmp_path):
        """Setup temporary database for testing"""
        # Create sample CSV
        sample_facts = pd.DataFrame({
            'statement': [
                "India has 28 states and 8 union territories",
                "The Earth revolves around the Sun",
                "Water boils at 100 degrees Celsius"
            ]
        })
        
        csv_path = tmp_path / "test_facts.csv"
        sample_facts.to_csv(csv_path, index=False)
        
        # Build test database
        with patch('config.FACTS_CSV_PATH', csv_path):
            with patch('config.VECTOR_INDEX_PATH', tmp_path / "test_index.bin"):
                vector_db.index_path = tmp_path / "test_index.bin"
                vector_db.build_and_save(sample_facts['statement'].tolist())
                vector_db.load()
    
    @patch('core.llm_service.llm_service.get_verdict')
    def test_pipeline_success(self, mock_verdict):
        """Test successful pipeline execution"""
        mock_verdict.return_value = Mock(
            verdict="True",
            confidence=0.95,
            reasoning="Test reasoning"
        )
        
        result = run_fact_checking_pipeline("India has 28 states")
        
        assert 'verdict' in result
        assert 'confidence' in result
        assert 'extracted_claim' in result
        assert 'evidence' in result
        assert 'performance' in result
    
    def test_pipeline_empty_input(self):
        """Test pipeline with empty input"""
        result = run_fact_checking_pipeline("")
        # Should handle gracefully
        assert isinstance(result, dict)
    
    @patch('core.llm_service.llm_service.get_verdict')
    def test_pipeline_timing(self, mock_verdict):
        """Test that pipeline records timing"""
        mock_verdict.return_value = Mock(
            verdict="True",
            confidence=0.9,
            reasoning="Test"
        )
        
        result = run_fact_checking_pipeline("Test claim")
        
        assert 'performance' in result
        assert 'total_time' in result['performance']
        assert 'extraction_time' in result['performance']
