# core/claim_extractor.py
import spacy
import logging
import subprocess
import sys
import re

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
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            print("Model 'en_core_web_md' not found. Downloading...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_md"])
            self.nlp = spacy.load("en_core_web_md")
        logger.info("SpaCy model loaded successfully.")

    def _clean_text(self, text: str) -> str:
        """Removes common conversational fillers or instructional phrases."""
        # NEW: More robust regex for "For Example" and similar constructs
        text = re.sub(r'^(for example\s*:?\s*|e\.g\.\s*:?\s*|i\.e\.\s*:?\s*)', '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'\s*\(\s*for example\s*\)\s*', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*\[\s*e\.g\.\s*\]\s*', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*I think that\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*It is believed that\s*', '', text, flags=re.IGNORECASE)
        # Add more patterns as needed
        return text.strip()

    def extract(self, text: str) -> str:
        logger.info(f"Extracting claim from text: {text}")
        
        # Step 1: Clean the input text
        cleaned_text = self._clean_text(text)
        doc = self.nlp(cleaned_text)

        # Step 2: Prioritize short, atomic sentences as-is (after cleaning)
        if len(cleaned_text.split()) <= 10 and cleaned_text.strip(): # Ensure not empty after cleaning
            logger.info("Cleaned text is short. Returning as-is.")
            return cleaned_text

        # Step 3: Iterate through sentences to find the most likely claim
        best_sentence = ""
        max_len = 0
        for sent in doc.sents:
            # Simple check for declarative sentences and non-empty
            if sent.text.strip() and not (sent.text.endswith('?') or sent.text.endswith('!')):
                if len(sent.text.split()) > max_len:
                    best_sentence = sent.text.strip()
                    max_len = len(sent.text.split())
        
        # If no suitable sentence found, fallback to the cleaned text itself
        # This handles cases where SpaCy might not segment sentences well for short inputs
        if not best_sentence and cleaned_text:
            best_sentence = cleaned_text

        doc_sent = self.nlp(best_sentence) if best_sentence else doc # Use doc if best_sentence is empty


        # Function to get full phrase including modifiers/prepositions for a token's subtree
        def get_full_phrase(token):
            # Sort subtree tokens by index to maintain original word order
            return " ".join([t.text for t in sorted(token.subtree, key=lambda t: t.i)])

        # Prioritized pattern matching on the selected sentence
        for token in doc_sent:
            # Root verb for SVO or Copular
            if token.dep_ == "ROOT" and (token.pos_ == "VERB" or token.lemma_ in ("be", "seem", "appear", "become")):
                subjects = [get_full_phrase(child) for child in token.children 
                            if child.dep_ in ("nsubj", "nsubjpass")]
                
                objects_complements = []
                for child in token.children:
                    # Direct objects, nominal/adjectival complements, open clausal complements
                    if child.dep_ in ("dobj", "attr", "acomp", "oprd", "xcomp"):
                        objects_complements.append(get_full_phrase(child))
                    # For prepositions, get the entire prepositional phrase using get_full_phrase on the prep token itself
                    elif child.dep_ == "prep":
                        objects_complements.append(get_full_phrase(child)) 
                    # Consider other important clausal modifiers if they are part of the main assertion
                    elif child.dep_ in ("advcl", "csubj"): 
                        objects_complements.append(get_full_phrase(child))

                auxiliaries = [t.text for t in token.lefts if t.dep_ == "aux"]
                negations = [t.text for t in token.lefts if t.dep_ == "neg"]
                
                # Reconstruct verb phrase with auxiliaries and negations
                verb_part_tokens = [t.text for t in sorted(token.lefts, key=lambda t: t.i) if t.dep_ in ("aux", "neg")]
                verb_part_tokens.append(token.text if not verb_part_tokens else token.lemma_) # Use lemma if aux present
                verb_part = " ".join(verb_part_tokens)


                if subjects:
                    claim_parts = [subjects[0], verb_part] + [p for p in objects_complements if p.strip()]
                    claim = " ".join(filter(None, claim_parts)).replace("  ", " ").strip()
                    if claim:
                        logger.info(f"Claim extracted from SVO/Copular pattern: {claim}")
                        return claim

        # Fallback 1: Return the best declarative sentence
        if best_sentence:
            logger.info(f"No specific pattern found. Returning best declarative sentence: {best_sentence}")
            return best_sentence

        # Fallback 2: If no suitable claim, sentence, or cleaned text is found, return original text
        logger.warning("No suitable claim, sentence, or cleaned text found. Returning original text.")
        return text.strip() # Return original input if everything else fails

claim_extractor = ClaimExtractor()