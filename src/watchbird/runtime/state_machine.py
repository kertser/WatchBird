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
        window_size: int = 10,
        confidence_decay_threshold: int = 3
    ):
        """Initialize track state machine.

        Args:
            track_id: Track identifier
            t_accept: Minimum median score for FRIENDLY acceptance
            t_margin: Minimum margin between best and second-best
            t_timeout: Timeout in seconds before becoming ENEMY
            consistency_count: Minimum consistency frames required
            window_size: Temporal aggregation window size
            confidence_decay_threshold: Number of inconsistent frames before dropping FRIENDLY
        """
        self.track_id = track_id
        self.t_accept = t_accept
        self.t_margin = t_margin
        self.t_timeout = t_timeout
        self.consistency_count = consistency_count
        self.confidence_decay_threshold = confidence_decay_threshold

        self.state = TrackState.SUSPECT
        self.person_id: Optional[str] = None
        self.confidence: float = 0.0
        self.start_time = time.time()

        # Tracks consecutive frames with inconsistent detection while FRIENDLY
        self.inconsistent_frame_count: int = 0

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
                    self.inconsistent_frame_count = 0

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

        elif self.state == TrackState.FRIENDLY:
            # Continue comparing faces and update confidence
            person_id, metrics = self.aggregator.get_aggregated_decision(
                consistency_count=self.consistency_count
            )

            if person_id is not None and metrics:
                median_score = metrics.get("median_score", 0.0)
                median_margin = metrics.get("median_margin", 0.0)
                consistency = metrics.get("consistency", 0)

                # Check if detection is still consistent with same person
                if (
                    person_id == self.person_id and
                    median_score >= self.t_accept and
                    median_margin >= self.t_margin and
                    consistency >= self.consistency_count
                ):
                    # Continuous detection - update confidence (gain confidence)
                    # Use exponential moving average to smoothly increase confidence
                    self.confidence = max(self.confidence, median_score)
                    self.inconsistent_frame_count = 0

                    logger.debug(
                        f"Track {self.track_id} FRIENDLY maintained ({self.person_id}, "
                        f"conf={self.confidence:.3f}, margin={median_margin:.3f})"
                    )
                else:
                    # Detection inconsistent - increment counter
                    self.inconsistent_frame_count += 1

                    logger.debug(
                        f"Track {self.track_id} FRIENDLY inconsistent frame "
                        f"({self.inconsistent_frame_count}/{self.confidence_decay_threshold}): "
                        f"detected={person_id}, expected={self.person_id}"
                    )

                    # Check if we should degrade back to SUSPECT
                    if self.inconsistent_frame_count >= self.confidence_decay_threshold:
                        old_person = self.person_id
                        old_confidence = self.confidence

                        self.state = TrackState.SUSPECT
                        self.person_id = None
                        self.confidence = 0.0
                        self.inconsistent_frame_count = 0
                        self.start_time = time.time()  # Reset timeout
                        self.aggregator.clear()  # Clear buffer to start fresh

                        logger.info(
                            f"Track {self.track_id} FRIENDLY → SUSPECT (lost consistency: "
                            f"was {old_person} conf={old_confidence:.3f})"
                        )
                        return True
            else:
                # No valid detection - increment inconsistent counter
                self.inconsistent_frame_count += 1

                if self.inconsistent_frame_count >= self.confidence_decay_threshold:
                    old_person = self.person_id
                    old_confidence = self.confidence

                    self.state = TrackState.SUSPECT
                    self.person_id = None
                    self.confidence = 0.0
                    self.inconsistent_frame_count = 0
                    self.start_time = time.time()
                    self.aggregator.clear()

                    logger.info(
                        f"Track {self.track_id} FRIENDLY → SUSPECT (no detection: "
                        f"was {old_person} conf={old_confidence:.3f})"
                    )
                    return True

        elif self.state == TrackState.ENEMY:
            # Continue detection to avoid false positives
            # ENEMY can transition back to FRIENDLY if properly identified
            person_id, metrics = self.aggregator.get_aggregated_decision(
                consistency_count=self.consistency_count
            )

            if person_id is not None and metrics:
                median_score = metrics.get("median_score", 0.0)
                median_margin = metrics.get("median_margin", 0.0)
                consistency = metrics.get("consistency", 0)

                # Log decision criteria for debugging
                logger.debug(
                    f"Track {self.track_id} (ENEMY): person={person_id}, "
                    f"score={median_score:.3f} (need>={self.t_accept:.2f}), "
                    f"margin={median_margin:.3f} (need>={self.t_margin:.2f}), "
                    f"consistency={consistency} (need>={self.consistency_count})"
                )

                # Check for FRIENDLY transition (same criteria as from SUSPECT)
                if (
                    median_score >= self.t_accept and
                    median_margin >= self.t_margin and
                    consistency >= self.consistency_count
                ):
                    self.state = TrackState.FRIENDLY
                    self.person_id = person_id
                    self.confidence = median_score
                    self.inconsistent_frame_count = 0

                    logger.info(
                        f"Track {self.track_id} ENEMY → FRIENDLY ({person_id}, "
                        f"conf={median_score:.3f}, margin={median_margin:.3f}, "
                        f"consistency={consistency}) - false positive corrected!"
                    )
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
        """Check if state is terminal.

        FRIENDLY is not terminal - we continue comparing faces.
        ENEMY is not terminal - we continue detection to avoid false positives.
        A person marked as ENEMY can still be re-identified as FRIENDLY.

        Returns:
            True if in terminal state (currently always False)
        """
        return False  # Continue detection for all states

