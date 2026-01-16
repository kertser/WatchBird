"""Temporal aggregation for multi-frame decision making."""

import logging
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TemporalAggregator:
    """Temporal aggregation buffer for multi-frame decisions."""

    def __init__(self, window_size: int = 10):
        """Initialize temporal aggregator.

        Args:
            window_size: Number of frames to keep in rolling window
        """
        self.window_size = window_size
        self.buffer: deque = deque(maxlen=window_size)

    def add_observation(
        self,
        person_id: Optional[str],
        score: float,
        second_score: float,
        reliability: float
    ) -> None:
        """Add observation to buffer.

        Args:
            person_id: Best matching person ID (or None)
            score: Best matching score
            second_score: Second best matching score
            reliability: Combined reliability score
        """
        observation = {
            "person_id": person_id,
            "score": score,
            "second_score": second_score,
            "margin": score - second_score,
            "reliability": reliability
        }
        self.buffer.append(observation)

    def get_aggregated_decision(
        self,
        consistency_count: int = 6
    ) -> Tuple[Optional[str], Dict[str, float]]:
        """Get aggregated decision from buffer.

        Args:
            consistency_count: Minimum number of frames where best_id must win

        Returns:
            Tuple of (best_person_id, metrics)
            metrics contains: median_score, median_margin, consistency, avg_reliability
        """
        if len(self.buffer) == 0:
            return None, {}

        # Count person_id occurrences
        person_counts: Dict[str, int] = {}
        scores_by_person: Dict[str, list] = {}
        margins_by_person: Dict[str, list] = {}
        reliabilities = []

        for obs in self.buffer:
            person_id = obs["person_id"]
            if person_id is not None:
                person_counts[person_id] = person_counts.get(person_id, 0) + 1

                if person_id not in scores_by_person:
                    scores_by_person[person_id] = []
                    margins_by_person[person_id] = []

                scores_by_person[person_id].append(obs["score"])
                margins_by_person[person_id].append(obs["margin"])

            reliabilities.append(obs["reliability"])

        # Check if any person_id meets consistency requirement
        if not person_counts:
            return None, {
                "median_score": 0.0,
                "median_margin": 0.0,
                "consistency": 0,
                "avg_reliability": np.mean(reliabilities) if reliabilities else 0.0
            }

        # Get most consistent person_id
        best_person_id = max(person_counts.items(), key=lambda x: x[1])[0]
        consistency = person_counts[best_person_id]

        # Compute metrics for best person
        scores = scores_by_person[best_person_id]
        margins = margins_by_person[best_person_id]

        metrics = {
            "median_score": float(np.median(scores)),
            "median_margin": float(np.median(margins)),
            "consistency": consistency,
            "avg_reliability": float(np.mean(reliabilities))
        }

        # Check if consistency requirement is met
        if consistency >= consistency_count:
            return best_person_id, metrics
        else:
            return None, metrics

    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.buffer) >= self.window_size

