# core/vector_db.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from typing import List, Tuple

from config import EMBEDDING_MODEL, VECTOR_INDEX_PATH, FACTS_CSV_PATH

class VectorDB:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = None
        self.facts = []

    def load(self):
        """Loads the FAISS index and facts from disk."""
        if not VECTOR_INDEX_PATH.exists():
            raise FileNotFoundError(f"Vector index not found at {VECTOR_INDEX_PATH}. Please run build_database.py first.")
        self.index = faiss.read_index(str(VECTOR_INDEX_PATH))
        self.facts = pd.read_csv(FACTS_CSV_PATH)["statement"].tolist()
        print("VectorDB loaded successfully.")

    def build_and_save(self, facts: List[str]):
        """Builds the FAISS index from a list of facts and saves it."""
        print("Building vector index...")
        fact_embeddings = self.embedding_model.encode(facts, convert_to_numpy=True, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index = faiss.IndexIDMap(self.index)
        self.index.add_with_ids(fact_embeddings.astype('float32'), np.arange(len(facts)))
        
        faiss.write_index(self.index, str(VECTOR_INDEX_PATH))
        print(f"Index built with {len(facts)} facts and saved to {VECTOR_INDEX_PATH}")
        self.facts = facts

    def search(self, query: str, k: int, threshold: float) -> List[Tuple[str, float]]:
        """Searches the index for the top-k most similar facts with a confidence threshold."""
        if self.index is None:
            self.load()
            
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for i, dist in zip(indices[0], distances[0]):
            # L2 distance needs to be converted to a more intuitive similarity score
            # similarity = 1 / (1 + distance)
            similarity = 1 / (1 + dist)
            if i > -1:
                results.append((self.facts[i], similarity))
        
        return results

vector_db = VectorDB()