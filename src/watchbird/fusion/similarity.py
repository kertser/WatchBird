"""Similarity computation and multi-modal fusion."""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SimilarityFusion:
    """Multi-modal similarity fusion."""

    def __init__(self, eps: float = 1e-6):
        """Initialize similarity fusion.

        Args:
            eps: Small constant to avoid division by zero
        """
        self.eps = eps

    def fuse_single_frame(
        self,
        face_similarity: Optional[Tuple[float, str]] = None,
        face_reliability: float = 0.0,
        reid_similarity: Optional[Tuple[float, str]] = None,
        reid_reliability: float = 0.0,
        gait_similarity: Optional[Tuple[float, str]] = None,
        gait_reliability: float = 0.0
    ) -> Dict[str, Tuple[float, float]]:
        """Fuse similarities from multiple modalities for a single frame.

        Args:
            face_similarity: Tuple of (similarity, person_id) or None
            face_reliability: Face quality/reliability score
            reid_similarity: Tuple of (similarity, person_id) or None
            reid_reliability: ReID quality/reliability score
            gait_similarity: Tuple of (similarity, person_id) or None
            gait_reliability: Gait quality/reliability score

        Returns:
            Dict mapping person_id to (fused_score, total_reliability)
        """
        # Collect weighted scores per person
        person_scores: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        # Add face contribution
        if face_similarity is not None and face_reliability > 0:
            sim, person_id = face_similarity
            person_scores[person_id].append((face_reliability * sim, face_reliability))

        # Add ReID contribution
        if reid_similarity is not None and reid_reliability > 0:
            sim, person_id = reid_similarity
            person_scores[person_id].append((reid_reliability * sim, reid_reliability))

        # Add gait contribution
        if gait_similarity is not None and gait_reliability > 0:
            sim, person_id = gait_similarity
            person_scores[person_id].append((gait_reliability * sim, gait_reliability))

        # Compute fused scores
        fused_scores = {}
        for person_id, scores in person_scores.items():
            total_weighted_sim = sum(ws for ws, _ in scores)
            total_reliability = sum(r for _, r in scores)

            fused_score = total_weighted_sim / (total_reliability + self.eps)
            fused_scores[person_id] = (fused_score, total_reliability)

        return fused_scores

    def get_top_matches(
        self,
        similarities: np.ndarray,
        indices: np.ndarray,
        person_ids: List[str],
        k: int = 2
    ) -> List[Tuple[float, str]]:
        """Get top-k matches from FAISS search results.

        Args:
            similarities: Similarity scores [k]
            indices: Vector indices [k]
            person_ids: List mapping vec_id to person_id
            k: Number of matches to return

        Returns:
            List of (similarity, person_id) tuples
        """
        matches = []

        for i in range(min(k, len(similarities))):
            sim = float(similarities[i])
            vec_id = int(indices[i])

            if vec_id < len(person_ids):
                person_id = person_ids[vec_id]
                matches.append((sim, person_id))

        return matches

    def aggregate_top_candidate(
        self,
        matches: List[Tuple[float, str]]
    ) -> Tuple[Optional[str], float, float]:
        """Aggregate matches to get best candidate and margin.

        Args:
            matches: List of (similarity, person_id) tuples

        Returns:
            Tuple of (best_person_id, best_score, margin)
            margin = best_score - second_best_score
        """
        if not matches:
            return None, 0.0, 0.0

        # Group by person_id and take max similarity
        person_best: Dict[str, float] = {}
        for sim, person_id in matches:
            person_best[person_id] = max(person_best.get(person_id, 0.0), sim)

        # Sort by score
        sorted_persons = sorted(person_best.items(), key=lambda x: x[1], reverse=True)

        best_person_id = sorted_persons[0][0]
        best_score = sorted_persons[0][1]

        second_score = sorted_persons[1][1] if len(sorted_persons) > 1 else 0.0
        margin = best_score - second_score

        return best_person_id, best_score, margin

