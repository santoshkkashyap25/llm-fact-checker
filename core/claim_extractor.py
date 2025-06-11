# core/claim_extractor.py
import spacy
import logging

# Configure logging to write to a file
logging.basicConfig(
    filename="claim_extractor.log",
    filemode="a",  # Append mode
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ClaimExtractor:
    def __init__(self):
        logger.info("Loading SpaCy model 'en_core_web_md'...")
        self.nlp = spacy.load("en_core_web_md")
        logger.info("SpaCy model loaded successfully.")

    def extract(self, text: str) -> str:
        """
        Extracts the main claim from a text using dependency parsing.
        Finds the root verb and its direct subject and object.
        """
        logger.info(f"Extracting claim from text: {text}")
        doc = self.nlp(text)
        
        for token in doc:
            # Find the main action/verb of the sentence
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subjects = [child.text for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                objects = [child.text for child in token.children if child.dep_ in ("dobj", "attr", "acomp")]
                
                if subjects and objects:
                    # Reconstruct a concise claim
                    claim = f"{' '.join(subjects)} {token.lemma_} {' '.join(objects)}"
                    logger.info(f"Claim extracted using subject-verb-object pattern: {claim}")
                    return claim

        # Fallback for simpler sentences or different structures
        if doc.noun_chunks:
            fallback_claim = max(doc.noun_chunks, key=len).text.strip()
            logger.info(f"No SVO pattern found. Using fallback noun chunk: {fallback_claim}")
            return fallback_claim
            
        logger.warning("No suitable claim found. Returning original text.")
        return text.strip()

claim_extractor = ClaimExtractor()


# # Sample inputs
# examples = [
#     "The company launched a new product last week.",
#     "Climate change is a serious threat to humanity.",
#     "Apple announced its earnings report.",
#     "There was a loud noise in the street.",
#     "Innovation drives progress."
# ]

# # Run extraction
# for i, text in enumerate(examples, 1):
#     claim = claim_extractor.extract(text)
#     print(f"Example {i}:")
#     print(f"Input: {text}")
#     print(f"Extracted Claim: {claim}")
#     print()
