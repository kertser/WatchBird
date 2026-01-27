"""Embedding aggregation for robust face recognition.

Collects multiple embeddings per track and computes a quality-weighted
centroid for matching, instead of noisy single-frame comparisons.

Flow:
    Frame → Embedding → Quality Check → Outlier Check → Buffer
                                                          ↓
    Buffer (20 max) → Quality² Weighted Average → Centroid
"""

import logging
from collections import deque
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingAggregator:
    """Per-track embedding aggregator with quality weighting and outlier filtering."""

    def __init__(
        self,
        window_size: int = 15,
        min_embeddings: int = 5,
        quality_threshold: float = 0.4,
        outlier_threshold: float = 0.25,
        use_quality_weighting: bool = True
    ):
        """Initialize embedding aggregator.

        Args:
            window_size: Maximum embeddings to keep in buffer
            min_embeddings: Minimum embeddings required before aggregation
            quality_threshold: Minimum quality score for embedding to be added
            outlier_threshold: Max distance from centroid to be included (0-1)
            use_quality_weighting: Weight embeddings by quality in centroid
        """
        self.window_size = window_size
        self.min_embeddings = min_embeddings
        self.quality_threshold = quality_threshold
        self.outlier_threshold = outlier_threshold
        self.use_quality_weighting = use_quality_weighting

        # Buffer of embedding data
        self.buffer: deque = deque(maxlen=window_size)

        # Cache for aggregated embedding (invalidated on add)
        self._cached_centroid: Optional[np.ndarray] = None
        self._cached_robust_centroid: Optional[np.ndarray] = None
        self._cache_valid = False

        # Track best quality embedding seen
        self._best_quality = 0.0
        self._best_embedding: Optional[np.ndarray] = None

    def add_embedding(
        self,
        embedding: np.ndarray,
        quality: float = 1.0,
        blur_score: float = 1.0,
        brightness: float = 0.5,
        face_size: int = 100
    ) -> bool:
        """Add an embedding to the buffer.

        Args:
            embedding: Face embedding vector
            quality: Overall quality score (0-1, from face detector confidence)
            blur_score: Blur quality (0-1, higher = sharper)
            brightness: Image brightness (0-1, 0.5 = optimal)
            face_size: Face bounding box size in pixels

        Returns:
            True if embedding was added, False if rejected
        """
        # Compute combined quality score
        combined_quality = self._compute_combined_quality(
            quality, blur_score, brightness, face_size
        )

        if combined_quality < self.quality_threshold:
            logger.debug(
                f"Embedding rejected: combined quality {combined_quality:.3f} < {self.quality_threshold}"
            )
            return False

        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm < 1e-6:
            logger.debug("Embedding rejected: zero norm")
            return False

        normalized = embedding / norm

        # Check for outlier if we have existing embeddings
        if len(self.buffer) >= 3:
            if self._is_outlier(normalized):
                logger.debug("Embedding rejected: outlier detected")
                return False

        self.buffer.append({
            'embedding': normalized,
            'quality': quality,
            'combined_quality': combined_quality,
            'blur_score': blur_score,
            'brightness': brightness,
            'face_size': face_size
        })

        # Track best quality embedding
        if combined_quality > self._best_quality:
            self._best_quality = combined_quality
            self._best_embedding = normalized.copy()

        # Invalidate cache
        self._cache_valid = False

        return True

    def _compute_combined_quality(
        self,
        quality: float,
        blur_score: float,
        brightness: float,
        face_size: int
    ) -> float:
        """Compute combined quality score from multiple factors.

        Args:
            quality: Base quality from detector
            blur_score: Blur detection score
            brightness: Image brightness
            face_size: Face size in pixels

        Returns:
            Combined quality score (0-1)
        """
        # Base quality (detection confidence)
        base_weight = 0.4

        # Blur penalty (blurry images get lower weight)
        blur_weight = 0.25
        blur_factor = blur_score  # Already 0-1

        # Brightness penalty (too dark or too bright is bad)
        # Optimal brightness is around 0.5
        brightness_weight = 0.15
        brightness_factor = 1.0 - abs(brightness - 0.5) * 2  # 1.0 at 0.5, 0 at 0 or 1
        brightness_factor = max(0.0, brightness_factor)

        # Face size bonus (larger faces are more reliable)
        size_weight = 0.20
        # Normalize face size: 50px = 0.5, 100px = 0.75, 200px+ = 1.0
        size_factor = min(1.0, 0.5 + face_size / 400.0)

        combined = (
            base_weight * quality +
            blur_weight * blur_factor +
            brightness_weight * brightness_factor +
            size_weight * size_factor
        )

        return float(np.clip(combined, 0.0, 1.0))

    def _is_outlier(self, embedding: np.ndarray) -> bool:
        """Check if embedding is an outlier compared to buffer.

        An outlier is an embedding that is very different from the
        current centroid, possibly indicating a different person
        or severe quality issues.

        Args:
            embedding: Normalized embedding to check

        Returns:
            True if embedding is an outlier
        """
        if len(self.buffer) < 3:
            return False

        # Compute quick centroid of existing embeddings
        embeddings = np.array([item['embedding'] for item in self.buffer])
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        # Compute distance to centroid
        similarity = np.dot(embedding, centroid)
        distance = 1.0 - similarity

        # Also check consistency with recent embeddings
        recent_embeddings = embeddings[-5:]  # Last 5
        recent_similarities = np.dot(recent_embeddings, embedding)
        avg_recent_similarity = np.mean(recent_similarities)

        # Outlier if too far from centroid AND inconsistent with recent
        is_outlier = (
            distance > self.outlier_threshold and
            avg_recent_similarity < (1.0 - self.outlier_threshold)
        )

        if is_outlier:
            logger.debug(
                f"Outlier: distance={distance:.3f}, recent_sim={avg_recent_similarity:.3f}"
            )

        return is_outlier

    def get_centroid(self, weighted: bool = True, robust: bool = True) -> Optional[np.ndarray]:
        """Get the centroid (average) embedding.

        Args:
            weighted: If True, weight embeddings by their quality scores
            robust: If True, exclude outliers from centroid calculation

        Returns:
            Normalized centroid embedding, or None if not enough embeddings
        """
        if len(self.buffer) < self.min_embeddings:
            return None

        # Use cached value if valid
        if self._cache_valid:
            if robust and self._cached_robust_centroid is not None:
                return self._cached_robust_centroid
            elif not robust and self._cached_centroid is not None:
                return self._cached_centroid

        embeddings = []
        weights = []

        for item in self.buffer:
            embeddings.append(item['embedding'])
            weights.append(item['combined_quality'] if 'combined_quality' in item else item['quality'])

        embeddings = np.array(embeddings)
        weights = np.array(weights)

        if robust and len(embeddings) > 3:
            # Iterative outlier removal
            mask = self._compute_inlier_mask(embeddings, weights)
            embeddings = embeddings[mask]
            weights = weights[mask]

            if len(embeddings) < self.min_embeddings:
                # Not enough inliers, fall back to all embeddings
                embeddings = np.array([item['embedding'] for item in self.buffer])
                weights = np.array([
                    item.get('combined_quality', item['quality'])
                    for item in self.buffer
                ])

        if weighted and self.use_quality_weighting:
            # Square the weights to give more importance to high-quality embeddings
            weights = weights ** 2
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
        if robust:
            self._cached_robust_centroid = centroid
        else:
            self._cached_centroid = centroid
        self._cache_valid = True

        return centroid

    def _compute_inlier_mask(
        self,
        embeddings: np.ndarray,
        weights: np.ndarray,
        iterations: int = 2
    ) -> np.ndarray:
        """Compute mask of inlier embeddings using iterative filtering.

        Args:
            embeddings: Array of embeddings
            weights: Quality weights
            iterations: Number of filtering iterations

        Returns:
            Boolean mask of inliers
        """
        mask = np.ones(len(embeddings), dtype=bool)

        for _ in range(iterations):
            if np.sum(mask) < 3:
                break

            # Compute weighted centroid of current inliers
            current_embeddings = embeddings[mask]
            current_weights = weights[mask]
            current_weights = current_weights / np.sum(current_weights)

            centroid = np.average(current_embeddings, axis=0, weights=current_weights)
            centroid = centroid / np.linalg.norm(centroid)

            # Compute distances to centroid
            similarities = np.dot(embeddings, centroid)
            distances = 1.0 - similarities

            # Mark outliers
            # Use adaptive threshold based on distribution
            mean_dist = np.mean(distances[mask])
            std_dist = np.std(distances[mask])
            threshold = min(self.outlier_threshold, mean_dist + 2 * std_dist)

            mask = distances <= threshold

        return mask

    def get_best_embedding(self) -> Optional[np.ndarray]:
        """Get the highest quality embedding seen.

        Useful as a fallback or for comparison.

        Returns:
            Best quality embedding or None
        """
        return self._best_embedding

    def get_embedding_count(self) -> int:
        """Get number of embeddings in buffer."""
        return len(self.buffer)

    def is_ready(self) -> bool:
        """Check if aggregator has enough embeddings for reliable matching."""
        return len(self.buffer) >= self.min_embeddings

    def get_quality_stats(self) -> Dict[str, float]:
        """Get quality statistics of buffered embeddings."""
        if len(self.buffer) == 0:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0, 'best': 0.0}

        qualities = [
            item.get('combined_quality', item['quality'])
            for item in self.buffer
        ]
        return {
            'mean': float(np.mean(qualities)),
            'min': float(np.min(qualities)),
            'max': float(np.max(qualities)),
            'count': len(qualities),
            'best': self._best_quality
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

    def get_consistency_score(self) -> float:
        """Get a consistency score for the track.

        High consistency = embeddings are similar to each other
        Low consistency = embeddings vary a lot (unreliable track)

        Returns:
            Consistency score (0-1), higher is better
        """
        variance = self.get_embedding_variance()
        # Convert distance to consistency (inverse)
        consistency = 1.0 - min(1.0, variance * 2)
        return float(consistency)

    def clear(self) -> None:
        """Clear all buffered embeddings."""
        self.buffer.clear()
        self._cached_centroid = None
        self._cached_robust_centroid = None
        self._cache_valid = False
        self._best_quality = 0.0
        self._best_embedding = None


class TrackEmbeddingManager:
    """Manages embedding aggregators for multiple tracks.

    Each track gets its own EmbeddingAggregator to collect embeddings over time.
    """

    def __init__(
        self,
        window_size: int = 15,
        min_embeddings: int = 5,
        quality_threshold: float = 0.4,
        outlier_threshold: float = 0.25
    ):
        """Initialize track embedding manager.

        Args:
            window_size: Maximum embeddings per track
            min_embeddings: Minimum embeddings before matching
            quality_threshold: Minimum quality for embedding acceptance
            outlier_threshold: Max distance from centroid for inliers
        """
        self.window_size = window_size
        self.min_embeddings = min_embeddings
        self.quality_threshold = quality_threshold
        self.outlier_threshold = outlier_threshold

        self.aggregators: Dict[int, EmbeddingAggregator] = {}

    def add_embedding(
        self,
        track_id: int,
        embedding: np.ndarray,
        quality: float = 1.0,
        blur_score: float = 1.0,
        brightness: float = 0.5,
        face_size: int = 100
    ) -> bool:
        """Add embedding for a track.

        Args:
            track_id: Track identifier
            embedding: Face embedding
            quality: Detection confidence
            blur_score: Blur quality (0-1)
            brightness: Image brightness (0-1)
            face_size: Face size in pixels

        Returns:
            True if embedding was added
        """
        if track_id not in self.aggregators:
            self.aggregators[track_id] = EmbeddingAggregator(
                window_size=self.window_size,
                min_embeddings=self.min_embeddings,
                quality_threshold=self.quality_threshold,
                outlier_threshold=self.outlier_threshold
            )

        return self.aggregators[track_id].add_embedding(
            embedding, quality, blur_score, brightness, face_size
        )

    def get_centroid(self, track_id: int, robust: bool = True) -> Optional[np.ndarray]:
        """Get centroid embedding for a track.

        Args:
            track_id: Track identifier
            robust: Use outlier-filtered centroid

        Returns:
            Centroid embedding or None if not ready
        """
        if track_id not in self.aggregators:
            return None

        return self.aggregators[track_id].get_centroid(robust=robust)

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

    def get_consistency(self, track_id: int) -> float:
        """Get consistency score for a track."""
        if track_id not in self.aggregators:
            return 0.0

        return self.aggregators[track_id].get_consistency_score()

    def get_quality_stats(self, track_id: int) -> Dict[str, float]:
        """Get quality statistics for a track."""
        if track_id not in self.aggregators:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0, 'best': 0.0}

        return self.aggregators[track_id].get_quality_stats()

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
