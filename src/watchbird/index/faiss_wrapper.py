"""FAISS index wrapper for similarity search."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissIndex:
    """FAISS index wrapper for vector similarity search."""

    def __init__(self, embedding_dim: int = 128):
        """Initialize FAISS index.

        Args:
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim
        self.index: Optional[faiss.Index] = None
        self.num_vectors = 0

    def build(self, embeddings: np.ndarray) -> bool:
        """Build FAISS index from embeddings.

        Args:
            embeddings: Array of embeddings [N, embedding_dim]

        Returns:
            True if successful, False otherwise
        """
        try:
            if len(embeddings.shape) != 2:
                logger.error(f"Expected 2D array, got shape {embeddings.shape}")
                return False

            if embeddings.shape[1] != self.embedding_dim:
                logger.error(
                    f"Embedding dim mismatch: expected {self.embedding_dim}, "
                    f"got {embeddings.shape[1]}"
                )
                return False

            # L2 normalize embeddings for cosine similarity
            embeddings = self._l2_normalize(embeddings)

            # Create IndexFlatIP (Inner Product) for cosine similarity
            self.index = faiss.IndexFlatIP(self.embedding_dim)

            # Add embeddings
            self.index.add(embeddings.astype(np.float32))
            self.num_vectors = embeddings.shape[0]

            logger.info(f"Built FAISS index with {self.num_vectors} vectors")
            return True

        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            return False

    def search(
        self,
        query: np.ndarray,
        k: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors.

        Args:
            query: Query embedding [embedding_dim] or [N, embedding_dim]
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (similarities, indices)
            - similarities: Cosine similarity scores [N, k]
            - indices: Vector indices [N, k]
        """
        if self.index is None:
            logger.warning("Index not built, returning empty results")
            return np.array([]), np.array([])

        try:
            # Ensure query is 2D
            if len(query.shape) == 1:
                query = query.reshape(1, -1)

            # L2 normalize query
            query = self._l2_normalize(query)

            # Limit k to available vectors
            k = min(k, self.num_vectors)

            # Search
            similarities, indices = self.index.search(query.astype(np.float32), k)

            return similarities, indices

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return np.array([]), np.array([])

    def save(self, path: str) -> bool:
        """Save FAISS index to file.

        Args:
            path: Path to save index

        Returns:
            True if successful, False otherwise
        """
        if self.index is None:
            logger.error("No index to save")
            return False

        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(path))
            logger.info(f"Saved FAISS index to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load FAISS index from file.

        Args:
            path: Path to index file

        Returns:
            True if successful, False otherwise
        """
        if not Path(path).exists():
            logger.error(f"Index file not found: {path}")
            return False

        try:
            self.index = faiss.read_index(str(path))
            self.num_vectors = self.index.ntotal
            logger.info(f"Loaded FAISS index from {path} ({self.num_vectors} vectors)")
            return True

        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False

    @staticmethod
    def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings.

        Args:
            embeddings: Embeddings array

        Returns:
            L2-normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        return embeddings / norms

