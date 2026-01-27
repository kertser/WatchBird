"""Embedding aggregation for robust face recognition.

This module implements embedding-level aggregation, which is more robust than
score-level aggregation. Instead of comparing each frame's embedding individually,
we collect multiple embeddings and compare their centroid (average) against the database.

This approach:
1. Reduces noise from single-frame variations
2. Provides more stable identity decisions
3. Better handles pose/lighting variations within a track
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingAggregator:
    """Aggregates embeddings over time for more robust recognition.

    Instead of making decisions on individual frame embeddings, this class:
    1. Collects embeddings over a sliding window
    2. Computes the centroid (normalized average) of collected embeddings
    3. Provides the aggregated embedding for matching

    This is similar to speaker verification systems that aggregate multiple
    audio frames before making a verification decision.
    """

    def __init__(
        self,
        window_size: int = 10,
        min_embeddings: int = 5,
        quality_threshold: float = 0.5
    ):
        """Initialize embedding aggregator.

        Args:
            window_size: Maximum embeddings to keep in buffer
            min_embeddings: Minimum embeddings required before aggregation
            quality_threshold: Minimum quality score for embedding to be added
        """
        self.window_size = window_size
        self.min_embeddings = min_embeddings
        self.quality_threshold = quality_threshold

        # Buffer of (embedding, quality) tuples
        self.buffer: deque = deque(maxlen=window_size)

        # Cache for aggregated embedding (invalidated on add)
        self._cached_centroid: Optional[np.ndarray] = None
        self._cache_valid = False

    def add_embedding(
        self,
        embedding: np.ndarray,
        quality: float = 1.0
    ) -> bool:
        """Add an embedding to the buffer.

        Args:
            embedding: Face embedding vector
            quality: Quality score for this embedding (0-1)

        Returns:
            True if embedding was added, False if rejected (low quality)
        """
        if quality < self.quality_threshold:
            logger.debug(f"Embedding rejected: quality {quality:.3f} < {self.quality_threshold}")
            return False

        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm < 1e-6:
            logger.debug("Embedding rejected: zero norm")
            return False

        normalized = embedding / norm

        self.buffer.append({
            'embedding': normalized,
            'quality': quality
        })

        # Invalidate cache
        self._cache_valid = False

        return True

    def get_centroid(self, weighted: bool = True) -> Optional[np.ndarray]:
        """Get the centroid (average) embedding.

        Args:
            weighted: If True, weight embeddings by their quality scores

        Returns:
            Normalized centroid embedding, or None if not enough embeddings
        """
        if len(self.buffer) < self.min_embeddings:
            return None

        # Use cached value if valid
        if self._cache_valid and self._cached_centroid is not None:
            return self._cached_centroid

        embeddings = []
        weights = []

        for item in self.buffer:
            embeddings.append(item['embedding'])
            weights.append(item['quality'])

        embeddings = np.array(embeddings)
        weights = np.array(weights)

        if weighted:
            # Normalize weights
            weights = weights / np.sum(weights)
            # Weighted average
            centroid = np.average(embeddings, axis=0, weights=weights)
        else:
            # Simple average
            centroid = np.mean(embeddings, axis=0)

        # Normalize the centroid
        norm = np.linalg.norm(centroid)
        if norm > 1e-6:
            centroid = centroid / norm

        # Cache the result
        self._cached_centroid = centroid
        self._cache_valid = True

        return centroid

    def get_embedding_count(self) -> int:
        """Get number of embeddings in buffer."""
        return len(self.buffer)

    def is_ready(self) -> bool:
        """Check if aggregator has enough embeddings for reliable matching."""
        return len(self.buffer) >= self.min_embeddings

    def get_quality_stats(self) -> Dict[str, float]:
        """Get quality statistics of buffered embeddings."""
        if len(self.buffer) == 0:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}

        qualities = [item['quality'] for item in self.buffer]
        return {
            'mean': float(np.mean(qualities)),
            'min': float(np.min(qualities)),
            'max': float(np.max(qualities)),
            'count': len(qualities)
        }

    def get_embedding_variance(self) -> float:
        """Get variance of embeddings in buffer.

        Lower variance indicates more consistent/stable track.
        High variance may indicate identity confusion or poor quality.

        Returns:
            Average pairwise distance between embeddings
        """
        if len(self.buffer) < 2:
            return 0.0

        embeddings = np.array([item['embedding'] for item in self.buffer])

        # Compute pairwise cosine similarities
        # For normalized vectors, cosine similarity = dot product
        similarities = embeddings @ embeddings.T

        # Get upper triangle (excluding diagonal)
        n = len(embeddings)
        upper_triangle = similarities[np.triu_indices(n, k=1)]

        # Convert to distance (1 - similarity) and return mean
        mean_distance = 1.0 - np.mean(upper_triangle)

        return float(mean_distance)

    def clear(self) -> None:
        """Clear all buffered embeddings."""
        self.buffer.clear()
        self._cached_centroid = None
        self._cache_valid = False


class TrackEmbeddingManager:
    """Manages embedding aggregators for multiple tracks.

    Each track gets its own EmbeddingAggregator to collect embeddings over time.
    """

    def __init__(
        self,
        window_size: int = 15,
        min_embeddings: int = 5,
        quality_threshold: float = 0.5
    ):
        """Initialize track embedding manager.

        Args:
            window_size: Maximum embeddings per track
            min_embeddings: Minimum embeddings before matching
            quality_threshold: Minimum quality for embedding acceptance
        """
        self.window_size = window_size
        self.min_embeddings = min_embeddings
        self.quality_threshold = quality_threshold

        self.aggregators: Dict[int, EmbeddingAggregator] = {}

    def add_embedding(
        self,
        track_id: int,
        embedding: np.ndarray,
        quality: float = 1.0
    ) -> bool:
        """Add embedding for a track.

        Args:
            track_id: Track identifier
            embedding: Face embedding
            quality: Embedding quality score

        Returns:
            True if embedding was added
        """
        if track_id not in self.aggregators:
            self.aggregators[track_id] = EmbeddingAggregator(
                window_size=self.window_size,
                min_embeddings=self.min_embeddings,
                quality_threshold=self.quality_threshold
            )

        return self.aggregators[track_id].add_embedding(embedding, quality)

    def get_centroid(self, track_id: int) -> Optional[np.ndarray]:
        """Get centroid embedding for a track.

        Args:
            track_id: Track identifier

        Returns:
            Centroid embedding or None if not ready
        """
        if track_id not in self.aggregators:
            return None

        return self.aggregators[track_id].get_centroid()

    def is_ready(self, track_id: int) -> bool:
        """Check if track has enough embeddings for matching.

        Args:
            track_id: Track identifier

        Returns:
            True if track is ready for matching
        """
        if track_id not in self.aggregators:
            return False

        return self.aggregators[track_id].is_ready()

    def get_embedding_count(self, track_id: int) -> int:
        """Get number of embeddings for a track."""
        if track_id not in self.aggregators:
            return 0

        return self.aggregators[track_id].get_embedding_count()

    def get_variance(self, track_id: int) -> float:
        """Get embedding variance for a track."""
        if track_id not in self.aggregators:
            return 0.0

        return self.aggregators[track_id].get_embedding_variance()

    def remove_track(self, track_id: int) -> None:
        """Remove a track's aggregator."""
        if track_id in self.aggregators:
            del self.aggregators[track_id]

    def cleanup_stale_tracks(self, active_track_ids: set) -> None:
        """Remove aggregators for tracks that are no longer active.

        Args:
            active_track_ids: Set of currently active track IDs
        """
        stale_ids = set(self.aggregators.keys()) - active_track_ids
        for track_id in stale_ids:
            del self.aggregators[track_id]
