# core/llm_service.py
import os
import logging
import re
from typing import List, Tuple
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEndpoint
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv  
from config import LLM_REPO_ID

# Configure logging
logging.basicConfig(
    filename="llm_service.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Verdict(BaseModel):
    verdict: str = Field(description="The final verdict, must be one of: 'True', 'False', 'Unverifiable'")
    confidence: float = Field(description="A confidence score from 0.0 to 1.0 for the verdict.")
    reasoning: str = Field(description="A detailed explanation for the verdict based on the evidence.")

class LLMService:
    def __init__(self):
        load_dotenv()
        if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
            raise ValueError("Hugging Face API token not found in environment variables.")
        
        logger.info("Initializing HuggingFace LLM endpoint...")
        self.llm = HuggingFaceEndpoint(
            repo_id=LLM_REPO_ID,
            temperature=0.1,
            max_new_tokens=512,
        )
        logger.info("LLM endpoint initialized.")

        self.parser = PydanticOutputParser(pydantic_object=Verdict)
        self.prompt = self._create_prompt_template()
        logger.info("Prompt template and output parser set up.")

    def _create_prompt_template(self) -> PromptTemplate:
        template = """
        SYSTEM: You are a strict fact-checking AI that performs precise evidence matching.
        Follow these rules absolutely:
        1. 'True' ONLY if evidence contains the EXACT factual content of the claim
        2. 'False' ONLY if evidence explicitly contradicts specific facts
        3. 'Unverifiable' if neither case above applies
        
        USER_CLAIM: "{claim}"
        
        RELEVANT_EVIDENCE:
        {evidence}
        
        INSTRUCTIONS:
        1. Normalize both claim and evidence (ignore case, punctuation, number formats)
        2. For numbers: treat '2005 crore', '₹2,005 cr', 'Rs. 2005 crores' as equivalent
        3. For organizations: treat 'IREDA', 'Ireda', 'India's Renewable Energy Dev. Agency' as equivalent
        4. If normalized claim exists in normalized evidence → 'True'
        5. If evidence contradicts specific facts → 'False'
        6. Otherwise → 'Unverifiable'
        
        {format_instructions}
        
        RESPONSE (JSON only):
        """
        return PromptTemplate(
            template=template,
            input_variables=["claim", "evidence"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase
        text = text.lower()
        # Standardize number formats
        text = re.sub(r'rs\.|₹|,', '', text)  # Remove currency symbols and commas
        text = re.sub(r'(\d+)\s*(crore|cr|crores)', r'\1 crore', text)  # Standardize crore
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # Standardize common entities
        text = re.sub(r'\bireda\b', 'india renewable energy development agency', text)
        # Remove extra spaces
        text = ' '.join(text.split())
        return text

    def _find_matches(self, claim: str, evidence: List[str]) -> Tuple[bool, List[str]]:
        """Check for matches between claim and evidence"""
        norm_claim = self._normalize_text(claim)
        matching_evidence = []
        
        for item in evidence:
            norm_item = self._normalize_text(item)
            if norm_claim in norm_item:
                matching_evidence.append(item)
        
        return (len(matching_evidence) > 0, matching_evidence)

    def get_verdict(self, claim: str, evidence: List[str]) -> Verdict:
        """Get fact-checking verdict with improved matching"""
        logger.info(f"Claim: {claim}")
        
        # Check for matches using normalized text
        has_match, matches = self._find_matches(claim, evidence)
        
        if has_match:
            logger.info(f"Found {len(matches)} matching evidence items")
            return Verdict(
                verdict="True",
                confidence=1.0,
                reasoning=f"The claim is supported by matching evidence: {matches[0]}"
            )
        
        # If no matches found, use LLM for partial verification
        evidence_str = "\n".join([f"- {e}" for e in evidence])
        chain = self.prompt | self.llm | self.parser
        
        try:
            result = chain.invoke({"claim": claim, "evidence": evidence_str})
            logger.info(f"LLM verdict: {result.verdict} (confidence: {result.confidence})")
            return result
        except Exception as e:
            logger.error(f"Error: {e}")
            return Verdict(
                verdict="Unverifiable",
                confidence=0.0,
                reasoning=f"Verification failed: {str(e)}"
            )

llm_service = LLMService()