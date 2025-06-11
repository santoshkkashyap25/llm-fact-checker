# pipeline.py
import logging
from typing import Dict, List, Any
from core.claim_extractor import claim_extractor
from core.vector_db import vector_db
from core.llm_service import llm_service
from config import TOP_K_RESULTS, CONFIDENCE_THRESHOLD

# Configure logging
logging.basicConfig(
    filename='pipeline.log',
    filemode='a',  # Append mode
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_fact_checking_pipeline(raw_text: str) -> Dict[str, Any]:
    """
    Runs the full RAG pipeline from raw text to a final verdict.
    """
    logger.info("Pipeline started")
    logger.info(f"Raw input text: {raw_text}")

    try:
        # 1. Extract the core claim
        claim = claim_extractor.extract(raw_text)
        logger.info(f"Extracted claim: {claim}")
        
        # 2. Retrieve relevant facts from the VectorDB
        retrieved_evidence_with_scores = vector_db.search(
            query=claim,
            k=TOP_K_RESULTS,
            threshold=CONFIDENCE_THRESHOLD
        )
        retrieved_evidence = [e[0] for e in retrieved_evidence_with_scores]
        logger.info(f"Retrieved {len(retrieved_evidence)} pieces of evidence")

        # 3. Get the final verdict from the LLM
        verdict_obj = llm_service.get_verdict(claim, retrieved_evidence)
        logger.info(f"Verdict: {verdict_obj.verdict}")
        logger.info(f"Confidence: {verdict_obj.confidence:.2f}")
        logger.info(f"Reasoning: {verdict_obj.reasoning}")

        # 4. Assemble the final response
        response = {
            "input_text": raw_text,
            "extracted_claim": claim,
            "verdict": verdict_obj.verdict,
            "confidence": f"{verdict_obj.confidence:.2f}",
            "reasoning": verdict_obj.reasoning,
            "evidence": retrieved_evidence
        }

        logger.info("Pipeline completed successfully")
        return response

    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        raise
