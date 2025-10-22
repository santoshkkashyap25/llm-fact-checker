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


    def test_fallback_verification_high_similarity(self, llm_service_mock):
        """Fallback verification returns True when similarity > 0.7"""
        claim = "Apple launches iPhone"
        evidence = ["Apple launches iPhone 15 today"]
        result = llm_service_mock._fallback_verification(claim, evidence, "LLM error")
        assert result.verdict == "True"
        assert result.confidence > 0.7
        assert "High similarity" in result.reasoning

    def test_fallback_verification_partial_similarity(self, llm_service_mock):
        """Fallback returns Unverifiable with moderate similarity"""
        claim = "Apple launches iPhone"
        evidence = ["Apple releases a new product line"]
        result = llm_service_mock._fallback_verification(claim, evidence, "LLM error")
        assert result.verdict == "Unverifiable"
        assert result.confidence == 0.5
        assert "Partial match" in result.reasoning

    def test_fallback_verification_low_similarity(self, llm_service_mock):
        """Fallback returns Unverifiable with low similarity"""
        claim = "Mars has water"
        evidence = ["Venus is the hottest planet"]
        result = llm_service_mock._fallback_verification(claim, evidence, "LLM error")
        assert result.verdict == "Unverifiable"
        assert result.confidence == 0.3
        assert "Low similarity" in result.reasoning


    def test_check_contradiction_none(self, llm_service_mock):
        """Contradiction check returns None when no contradiction"""
        claim = "India has 28 states"
        evidence = ["India consists of 28 states and 8 union territories"]
        result = llm_service_mock._check_contradiction(claim, evidence)
        assert result is None

    def test_cache_usage(self, llm_service_mock):
        """Cached result is returned instead of calling LLM again"""
        claim = "Test caching"
        evidence = ["Evidence caching test"]
        # First call to populate cache
        first_result = llm_service_mock.get_verdict(claim, evidence)
        # Mock LLM to raise exception if called again
        llm_service_mock.llm.chat_completion.side_effect = Exception("Should not be called")
        second_result = llm_service_mock.get_verdict(claim, evidence)
        assert first_result.dict() == second_result.dict()  # Cached result returned

    def test_normalize_text_entities(self, llm_service_mock):
        """Normalization handles entity replacement"""
        text = "IREDA invested Rs. 5,000 crore"
        norm_text = llm_service_mock._normalize_text(text)
        assert "india renewable energy development agency" in norm_text
        assert "5000" in norm_text