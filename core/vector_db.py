# core/vector_db.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import logging
from config import (
    EMBEDDING_MODEL, VECTOR_INDEX_PATH, 
    FACTS_CSV_PATH, DATA_DIR
)

logger = logging.getLogger(__name__)

class VectorDB:
    """Enhanced FAISS vector database with metadata support"""
    
    def __init__(self):
        self.embedding_model = None
        self.embedding_dim = None
        self.index = None
        self.facts = []
        self.metadata = []
        self.index_path = VECTOR_INDEX_PATH
    
    def _initialize_model(self):
        """Lazy load embedding model"""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Dimension: {self.embedding_dim}")
    
    def load(self):
        """Load FAISS index and facts from disk"""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Vector index not found at {self.index_path}. "
                f"Please run 'python build_database.py' first."
            )
        
        self._initialize_model()
        
        try:
            self.index = faiss.read_index(str(self.index_path))
            
            # Load facts and metadata
            df = pd.read_csv(FACTS_CSV_PATH)
            self.facts = df["statement"].tolist()
            self.metadata = df.to_dict('records') if len(df.columns) > 1 else None
            
            logger.info(f"VectorDB loaded: {len(self.facts)} facts indexed")
            
        except Exception as e:
            logger.error(f"Error loading vector DB: {e}")
            raise
    
    def build_and_save(self, facts: List[str], metadata: Optional[List[Dict]] = None):
        """Build FAISS index from facts and save to disk"""
        self._initialize_model()
        
        logger.info(f"Building index for {len(facts)} facts...")
        
        # Generate embeddings with progress
        fact_embeddings = self.embedding_model.encode(
            facts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index = faiss.IndexIDMap(self.index)
        
        # Add vectors with IDs
        ids = np.arange(len(facts))
        self.index.add_with_ids(fact_embeddings.astype('float32'), ids)
        
        # Save index
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        
        self.facts = facts
        self.metadata = metadata
        
        logger.info(f"✓ Index built and saved to {self.index_path}")
        logger.info(f"  - Total facts: {len(facts)}")
        logger.info(f"  - Dimension: {self.embedding_dim}")
        logger.info(f"  - Index type: {type(self.index).__name__}")
    
    def search(
        self,
        query: str,
        k: int,
        threshold: float
    ) -> List[Tuple[str, float, Optional[Dict]]]:
        """
        Search for similar facts with confidence threshold.
        Returns: List of (fact_text, similarity_score, metadata)
        """
        if self.index is None:
            self.load()
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search index
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            min(k, len(self.facts))  # Don't request more than we have
        )
        
        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i == -1:  # Invalid index
                continue
            
            # Convert L2 distance to similarity score
            similarity = 1 / (1 + dist)
            
            # Apply threshold
            if similarity >= threshold:
                fact = self.facts[i]
                meta = self.metadata[i] if self.metadata else None
                results.append((fact, similarity, meta))
        
        logger.info(f"Retrieved {len(results)} facts above threshold {threshold}")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if self.index is None:
            return {'status': 'not_loaded'}
        
        return {
            'status': 'loaded',
            'total_facts': len(self.facts),
            'embedding_dim': self.embedding_dim,
            'index_type': type(self.index).__name__,
            'has_metadata': self.metadata is not None
        }

vector_db = VectorDB()
