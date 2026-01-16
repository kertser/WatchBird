"""State machine for track classification."""

import logging
import time
from enum import Enum
from typing import Dict, Optional

from watchbird.fusion.aggregation import TemporalAggregator

logger = logging.getLogger(__name__)


class TrackState(Enum):
    """Track classification state."""
    SUSPECT = "SUSPECT"
    FRIENDLY = "FRIENDLY"
    ENEMY = "ENEMY"


class TrackStateMachine:
    """State machine for individual track classification."""

    def __init__(
        self,
        track_id: int,
        t_accept: float = 0.65,
        t_margin: float = 0.10,
        t_timeout: float = 5.0,
        consistency_count: int = 6,
        window_size: int = 10
    ):
        """Initialize track state machine.

        Args:
            track_id: Track identifier
            t_accept: Minimum median score for FRIENDLY acceptance
            t_margin: Minimum margin between best and second-best
            t_timeout: Timeout in seconds before becoming ENEMY
            consistency_count: Minimum consistency frames required
            window_size: Temporal aggregation window size
        """
        self.track_id = track_id
        self.t_accept = t_accept
        self.t_margin = t_margin
        self.t_timeout = t_timeout
        self.consistency_count = consistency_count

        self.state = TrackState.SUSPECT
        self.person_id: Optional[str] = None
        self.confidence: float = 0.0
        self.start_time = time.time()

        self.aggregator = TemporalAggregator(window_size=window_size)
        self.modalities_used: Dict[str, float] = {}

    def update(
        self,
        best_person_id: Optional[str],
        best_score: float,
        second_score: float,
        reliability: float,
        modality: str = "face"
    ) -> bool:
        """Update state machine with new observation.

        Args:
            best_person_id: Best matching person ID
            best_score: Best matching score
            second_score: Second best matching score
            reliability: Observation reliability
            modality: Modality type (face, reid, gait)

        Returns:
            True if state changed, False otherwise
        """
        # Update modalities used
        self.modalities_used[modality] = reliability

        # Add observation to aggregator
        self.aggregator.add_observation(
            best_person_id,
            best_score,
            second_score,
            reliability
        )

        # Check for state transition
        return self._check_transition()

    def _check_transition(self) -> bool:
        """Check and perform state transitions.

        Returns:
            True if state changed, False otherwise
        """
        if self.state == TrackState.SUSPECT:
            # Check for FRIENDLY transition
            person_id, metrics = self.aggregator.get_aggregated_decision(
                consistency_count=self.consistency_count
            )

            if person_id is not None and metrics:
                median_score = metrics.get("median_score", 0.0)
                median_margin = metrics.get("median_margin", 0.0)
                consistency = metrics.get("consistency", 0)

                # Log decision criteria for debugging
                logger.debug(
                    f"Track {self.track_id}: person={person_id}, "
                    f"score={median_score:.3f} (need>={self.t_accept:.2f}), "
                    f"margin={median_margin:.3f} (need>={self.t_margin:.2f}), "
                    f"consistency={consistency} (need>={self.consistency_count})"
                )

                # Check acceptance criteria
                if (
                    median_score >= self.t_accept and
                    median_margin >= self.t_margin and
                    consistency >= self.consistency_count
                ):
                    self.state = TrackState.FRIENDLY
                    self.person_id = person_id
                    self.confidence = median_score

                    logger.info(
                        f"Track {self.track_id} → FRIENDLY ({person_id}, "
                        f"conf={median_score:.3f}, margin={median_margin:.3f}, "
                        f"consistency={consistency})"
                    )
                    return True
                else:
                    # Log why it didn't pass
                    reasons = []
                    if median_score < self.t_accept:
                        reasons.append(f"score too low ({median_score:.3f}<{self.t_accept:.2f})")
                    if median_margin < self.t_margin:
                        reasons.append(f"margin too low ({median_margin:.3f}<{self.t_margin:.2f})")
                    if consistency < self.consistency_count:
                        reasons.append(f"inconsistent ({consistency}<{self.consistency_count})")

                    # Log at INFO level when approaching timeout to help diagnose
                    elapsed = time.time() - self.start_time
                    if elapsed > self.t_timeout * 0.75:  # Last 25% of timeout period
                        logger.info(
                            f"Track {self.track_id} approaching timeout ({elapsed:.1f}s): "
                            f"person={person_id}, score={median_score:.3f}, "
                            f"margin={median_margin:.3f}, consistency={consistency}/{self.consistency_count} - "
                            f"{', '.join(reasons)}"
                        )
                    else:
                        logger.debug(
                            f"Track {self.track_id}: NOT FRIENDLY yet - {', '.join(reasons)}"
                        )
                    # Don't return here - need to check timeout below!

            # Check for ENEMY transition (timeout)
            elapsed = time.time() - self.start_time
            if elapsed > self.t_timeout:
                self.state = TrackState.ENEMY
                logger.info(f"Track {self.track_id} → ENEMY (timeout after {elapsed:.1f}s)")
                return True

        return False

    def get_event(self) -> Dict:
        """Get current state as event dict.

        Returns:
            Event dictionary
        """
        event = {
            "ts": time.time(),
            "track_id": self.track_id,
            "state": self.state.value,
            "person_id": self.person_id,
            "confidence": self.confidence,
            "modalities_used": self.modalities_used.copy()
        }
        return event

    def is_terminal(self) -> bool:
        """Check if state is terminal (FRIENDLY or ENEMY).

        Returns:
            True if in terminal state
        """
        return self.state in (TrackState.FRIENDLY, TrackState.ENEMY)

