import pytest
from core.llm_service import LLMService, Verdict
from unittest.mock import Mock, patch

class TestLLMService:
    """Test suite for LLM service"""

    @pytest.fixture
    def llm_service_mock(self):
        """Mock LLM service to avoid real API calls"""
        with patch('core.llm_service.InferenceClient') as mock_client:
            mock_instance = Mock()
            # Mock the chat_completion method
            mock_instance.chat_completion.return_value = Mock(
                choices=[Mock(message=Mock(content='{"verdict": "True", "confidence": 0.9, "reasoning": "Test reasoning"}'))]
            )
            mock_client.return_value = mock_instance
            service = LLMService()
            return service

    def test_normalize_text(self, llm_service_mock):
        """Test text normalization"""
        text1 = "Rs. 2,005 crores"
        text2 = "₹2005 crore"
        norm1 = llm_service_mock._normalize_text(text1)
        norm2 = llm_service_mock._normalize_text(text2)
        assert "2005" in norm1
        assert "2005" in norm2

    def test_exact_match_detection(self, llm_service_mock):
        """Test exact match detection"""
        claim = "India has 28 states"
        evidence = [
            "India consists of 28 states and 8 union territories",
            "The capital of India is New Delhi"
        ]
        has_match, matches = llm_service_mock._check_exact_match(claim, evidence)
        assert has_match
        assert len(matches) > 0

    def test_no_match(self, llm_service_mock):
        """Test when no match exists"""
        claim = "Mars is red"
        evidence = [
            "Venus is the hottest planet",
            "Jupiter is the largest planet"
        ]
        has_match, matches = llm_service_mock._check_exact_match(claim, evidence)
        assert not has_match
        assert len(matches) == 0

    def test_get_verdict_llm(self, llm_service_mock):
        """Test LLM verdict parsing"""
        claim = "Test claim"
        evidence = ["Test evidence"]
        verdict = llm_service_mock.get_verdict(claim, evidence)
        assert isinstance(verdict, Verdict)
        assert verdict.verdict in ["True", "False", "Unverifiable"]
        assert 0.0 <= verdict.confidence <= 1.0
        assert isinstance(verdict.reasoning, str)
