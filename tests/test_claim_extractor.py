import pytest
from core.claim_extractor import claim_extractor

class TestClaimExtractor:
    """Test suite for claim extraction"""
    
    def test_simple_declarative(self):
        """Test extraction of simple declarative sentence"""
        text = "The Earth is round."
        claim = claim_extractor.extract(text)
        assert "earth" in claim.lower()
        assert "round" in claim.lower()
    
    def test_compound_sentence(self):
        """Test extraction from compound sentence"""
        text = "Scientists say climate change is real, but some deny it."
        claim = claim_extractor.extract(text)
        assert len(claim) > 0
        assert "climate change" in claim.lower() or "real" in claim.lower()
    
    def test_with_conversational_filler(self):
        """Test removal of conversational fillers"""
        text = "For example, India has 28 states."
        claim = claim_extractor.extract(text)
        assert "for example" not in claim.lower()
        assert "india" in claim.lower()
        assert "28" in claim or "states" in claim.lower()
    
    def test_question_handling(self):
        """Test handling of questions"""
        text = "Is the Eiffel Tower in Paris?"
        claim = claim_extractor.extract(text)
        # Should still extract something meaningful
        assert len(claim) > 0
    
    def test_already_atomic(self):
        """Test that atomic claims are returned as-is"""
        text = "India has 28 states"
        claim = claim_extractor.extract(text)
        assert "india" in claim.lower()
        assert "states" in claim.lower()
    
    def test_empty_input(self):
        """Test empty input handling"""
        text = ""
        claim = claim_extractor.extract(text)
        assert claim == ""
    
    def test_very_long_text(self):
        """Test extraction from long text"""
        text = (
            "According to recent reports, which were published last week, "
            "India achieved a peak power demand of 241 GW on June 9, 2025. "
            "This is a significant milestone for the country."
        )
        claim = claim_extractor.extract(text)
        assert len(claim) > 10
        assert "india" in claim.lower() or "241" in claim or "gw" in claim.lower()
    
    @pytest.mark.parametrize("input_text,expected_phrase", [
        ("I think that vaccines are safe.", "vaccines"),
        ("It is believed that Mars has water.", "mars"),
        ("The capital of France is Paris.", "paris"),
    ])
    def test_multiple_patterns(self, input_text, expected_phrase):
        """Test various input patterns"""
        claim = claim_extractor.extract(input_text)
        assert expected_phrase.lower() in claim.lower()

