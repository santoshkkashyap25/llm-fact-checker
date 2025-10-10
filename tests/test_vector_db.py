import pytest
from core.vector_db import VectorDB
from config import FACTS_CSV_PATH, VECTOR_INDEX_PATH
import pandas as pd

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
