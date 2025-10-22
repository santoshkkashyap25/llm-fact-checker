import pytest
from core.vector_db import VectorDB
from config import FACTS_CSV_PATH, VECTOR_INDEX_PATH
import pandas as pd
from unittest.mock import patch, MagicMock
import numpy as np 

class TestVectorDB:
    """Test suite for vector database"""
    
    @pytest.fixture
    def sample_facts(self):
        """Sample facts for testing"""
        return [
            "India has 28 states and 8 union territories",
            "The Earth revolves around the Sun",
            "Water boils at 100 degrees Celsius",
            "Python is a programming language",
            "The Great Wall of China is visible from space"
        ]
    
    @pytest.fixture
    def temp_db(self, sample_facts, tmp_path):
        """Create temporary database for testing"""
        db = VectorDB()
        db.index_path = tmp_path / "test_index.bin"
        db.build_and_save(sample_facts)
        return db
    
    def test_build_and_save(self, temp_db, sample_facts):
        """Test database building"""
        assert temp_db.index is not None
        assert len(temp_db.facts) == len(sample_facts)
        assert temp_db.index_path.exists()
    
    def test_search_exact_match(self, temp_db):
        """Test search with exact match"""
        results = temp_db.search("India has 28 states", k=3, threshold=0.5)
        assert len(results) > 0
        # Should find the India fact
        assert any("india" in r[0].lower() for r in results)
    
    def test_search_semantic(self, temp_db):
        """Test semantic search"""
        results = temp_db.search("What is the boiling point of water?", k=3, threshold=0.4)
        # Should find water/boiling fact
        assert any("water" in r[0].lower() or "boil" in r[0].lower() for r in results)
    
    def test_search_with_threshold(self, temp_db):
        """Test that threshold filters results"""
        high_threshold = temp_db.search("Random query", k=5, threshold=0.9)
        low_threshold = temp_db.search("Random query", k=5, threshold=0.1)
        assert len(high_threshold) <= len(low_threshold)
    
    def test_empty_query(self, temp_db):
        """Test empty query handling"""
        results = temp_db.search("", k=3, threshold=0.5)
        # Should return something (even if low relevance)
        assert isinstance(results, list)


    @pytest.fixture
    def db(self, tmp_path, sample_facts):
        db = VectorDB()
        db.index_path = tmp_path / "test_index.bin"
        db.build_and_save(sample_facts)
        return db

    def test_get_stats_not_loaded(self):
        """Stats before DB load"""
        db = VectorDB()
        stats = db.get_stats()
        assert stats['status'] == 'not_loaded'

    def test_get_stats_loaded(self, db):
        """Stats after DB load"""
        stats = db.get_stats()
        assert stats['status'] == 'loaded'
        assert stats['total_facts'] == len(db.facts)
        assert stats['embedding_dim'] == db.embedding_dim


    def test_search_invalid_index_entry(self, db):
        """Handle invalid index IDs (-1) gracefully"""
        # Mock index search to return -1
        db.index.search = MagicMock(return_value=(np.array([[0.1]]), np.array([[-1]])))
        results = db.search("Test", k=1, threshold=0.0)
        assert results == []

    def test_build_and_save_with_metadata(self, tmp_path):
        """Test building index with metadata"""
        facts = ["Fact A", "Fact B"]
        metadata = [{"source": "A"}, {"source": "B"}]
        db = VectorDB()
        db.index_path = tmp_path / "index_meta.bin"
        db.build_and_save(facts, metadata=metadata)
        assert db.metadata == metadata
        assert len(db.facts) == 2

    def test_initialize_model_called_once(self, db):
        """_initialize_model should not reload model if already loaded"""
        model_before = db.embedding_model
        db._initialize_model()
        assert db.embedding_model is model_before

    def test_load_missing_index_file(self, tmp_path):
        """Load raises FileNotFoundError if index file missing"""
        db = VectorDB()
        db.index_path = tmp_path / "non_existent_index.bin"
        with pytest.raises(FileNotFoundError):
            db.load()

    @patch('core.vector_db.SentenceTransformer')
    def test_build_and_save_calls_model_encode(self, mock_model, tmp_path):
        """Ensure SentenceTransformer.encode is called during build"""
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 5
        mock_instance.encode.return_value = np.random.rand(2, 5)
        mock_model.return_value = mock_instance

        db = VectorDB()
        db.index_path = tmp_path / "mock_index.bin"
        facts = ["Fact1", "Fact2"]
        db.build_and_save(facts)
        mock_instance.encode.assert_called_once_with(
            facts, convert_to_numpy=True, show_progress_bar=True, batch_size=32
        )
