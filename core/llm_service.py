# core/llm_service.py
import os
import logging
from typing import List
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEndpoint
from langchain.output_parsers import PydanticOutputParser

from config import LLM_REPO_ID

# Configure logging to write to a file
logging.basicConfig(
    filename="llm_service.log",
    filemode="a",  # Append mode
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
        SYSTEM: You are a meticulous, unbiased AI fact-checker. Your task is to analyze a user's claim against a set of retrieved, verified facts.
        Your response MUST be a valid JSON object matching the provided format. Do not include any other text, greetings, or explanations outside of the JSON structure.

        USER_CLAIM: "{claim}"

        VERIFIED_EVIDENCE:
        {evidence}
        
        INSTRUCTIONS:
        1. Compare the USER_CLAIM to the VERIFIED_EVIDENCE.
        2. Determine if the evidence supports, contradicts, or is irrelevant to the claim.
        3. If the evidence directly supports the claim, classify as 'True'.
        4. If the evidence directly contradicts the claim, classify as 'False'.
        5. If the evidence is insufficient, unrelated, or the claim is too specific to be verified, classify as 'Unverifiable'.
        6. Provide a confidence score for your verdict (0.0 for Unverifiable, >0.5 for True/False).
        7. Write a concise, neutral reasoning.
        
        {format_instructions}
        
        AI_RESPONSE (JSON only):
        """
        return PromptTemplate(
            template=template,
            input_variables=["claim", "evidence"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def get_verdict(self, claim: str, evidence: List[str]) -> Verdict:
        """Invokes the LLM with the claim and evidence, and returns a parsed Verdict object."""
        logger.info(f"Received claim for verification: {claim}")
        
        if not evidence:
            logger.info("No evidence provided. Returning 'Unverifiable'.")
            return Verdict(
                verdict="🤷‍♂️ Unverifiable",
                confidence=0.0,
                reasoning="No relevant evidence was found in the trusted fact base to verify this claim."
            )
        
        evidence_str = "\n".join([f"- {e}" for e in evidence])
        logger.info(f"Formatted evidence:\n{evidence_str}")
        chain = self.prompt | self.llm | self.parser
        
        try:
            result = chain.invoke({"claim": claim, "evidence": evidence_str})
            logger.info(f"LLM returned verdict: {result.verdict}, confidence: {result.confidence}")
            return result
        except Exception as e:
            logger.error(f"Error invoking LLM or parsing response: {e}")
            return Verdict(
                verdict="Error",
                confidence=0.0,
                reasoning=f"An error occurred during verification: {e}"
            )

llm_service = LLMService()
