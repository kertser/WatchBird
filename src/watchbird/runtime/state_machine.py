"""State machine for track classification with cumulative confidence."""

import logging
import time
from enum import Enum
from typing import Dict, Optional

from watchbird.fusion.aggregation import TemporalAggregator

logger = logging.getLogger(__name__)


class TrackState(Enum):
    """Track classification state."""
    SUSPECT = "SUSPECT"       # Unknown, actively recognizing
    FRIENDLY = "FRIENDLY"     # Recognized, still building confidence
    CONFIRMED = "CONFIRMED"   # High confidence, tracking-only mode
    ENEMY = "ENEMY"           # Timed out without recognition


class TrackStateMachine:
    """State machine for individual track classification with cumulative confidence.

    State flow:
        SUSPECT → FRIENDLY → CONFIRMED
           ↓         ↓          ↓
         ENEMY    SUSPECT    FRIENDLY (if confidence drops)

    CONFIRMED state:
        - High cumulative confidence (>= confirm_threshold)
        - Recognition stops, only tracking
        - Track recovery if briefly lost
    """

    def __init__(
        self,
        track_id: int,
        t_accept: float = 0.65,
        t_margin: float = 0.10,
        t_timeout: float = 5.0,
        consistency_count: int = 6,
        window_size: int = 10,
        confidence_decay_threshold: int = 3,
        identity_switch_margin: float = 0.10,
        # New parameters for cumulative confidence
        confirm_threshold: float = 0.95,
        confidence_gain_rate: float = 0.05,
        confidence_decay_rate: float = 0.02,
        track_lost_timeout: float = 3.0
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
            identity_switch_margin: Extra margin required to switch to different person
            confirm_threshold: Confidence threshold to enter CONFIRMED state (e.g., 0.95)
            confidence_gain_rate: EMA rate for confidence gain per positive frame
            confidence_decay_rate: Rate of confidence decay per negative frame
            track_lost_timeout: Seconds before CONFIRMED drops to FRIENDLY if track lost
        """
        self.track_id = track_id
        self.t_accept = t_accept
        self.t_margin = t_margin
        self.t_timeout = t_timeout
        self.consistency_count = consistency_count
        self.confidence_decay_threshold = confidence_decay_threshold
        self.identity_switch_margin = identity_switch_margin

        # Cumulative confidence parameters
        self.confirm_threshold = confirm_threshold
        self.confidence_gain_rate = confidence_gain_rate
        self.confidence_decay_rate = confidence_decay_rate
        self.track_lost_timeout = track_lost_timeout

        self.state = TrackState.SUSPECT
        self.person_id: Optional[str] = None
        self.confidence: float = 0.0
        self.cumulative_confidence: float = 0.0  # Builds up over time
        self.start_time = time.time()

        # Tracks consecutive frames with inconsistent detection while FRIENDLY
        self.inconsistent_frame_count: int = 0

        # Track loss tracking for CONFIRMED state
        self.last_seen_time: float = time.time()
        self.frames_since_seen: int = 0

        # Locked identity - once identified, this person is "locked in"
        # and requires much stronger evidence to switch to a different person
        self.locked_person_id: Optional[str] = None
        self.locked_confidence: float = 0.0

        # Recognition skip flag for CONFIRMED state
        self.skip_recognition: bool = False

        # Recovery flag - prevents normal transitions from overwriting recovered state
        self._recovered: bool = False

        self.aggregator = TemporalAggregator(window_size=window_size)
        self.modalities_used: Dict[str, float] = {}

    def should_skip_recognition(self) -> bool:
        """Check if recognition should be skipped (CONFIRMED state).

        In CONFIRMED state, we only track - no embedding extraction/matching needed.
        This saves significant CPU/GPU resources.

        Returns:
            True if recognition should be skipped
        """
        return self.state == TrackState.CONFIRMED and self.skip_recognition

    def notify_track_seen(self) -> None:
        """Notify that the track is still being tracked (even without recognition).

        Call this every frame when the track is visible to maintain CONFIRMED state.
        """
        self.last_seen_time = time.time()
        self.frames_since_seen = 0

    def fast_recover(
        self,
        person_id: str,
        cumulative_confidence: float,
        min_confidence: float = 0.5
    ) -> bool:
        """Fast-recover a previously confirmed identity.

        Called when a new track matches a recently lost CONFIRMED/FRIENDLY track.
        Skips the full recognition process and directly enters CONFIRMED state
        to avoid the full recognition flow.

        Args:
            person_id: The recovered person ID
            cumulative_confidence: The cumulative confidence from the lost track
            min_confidence: Minimum confidence to go directly to CONFIRMED

        Returns:
            True if recovery successful
        """
        # Mark as recovered to prevent normal transitions from overwriting
        self._recovered = True

        # Restore identity
        self.person_id = person_id
        self.locked_person_id = person_id

        # Restore confidence (with slight reduction for safety)
        # But ensure we reach CONFIRMED threshold to avoid going through FRIENDLY again
        restored_cumulative = max(cumulative_confidence * 0.95, self.confirm_threshold)
        self.cumulative_confidence = min(1.0, restored_cumulative)
        self.confidence = max(0.8, cumulative_confidence)
        self.locked_confidence = self.confidence

        # Seed the aggregator with strong fake observations
        # Use high scores to ensure the FRIENDLY block conditions are met
        for _ in range(self.consistency_count + 5):
            self.aggregator.add_observation(
                person_id,
                self.confidence,  # best_score
                0.0,  # second_score (no competition = high margin)
                0.9   # reliability
            )

        # Always go directly to CONFIRMED for recovered tracks
        # This prevents the FRIENDLY block from potentially resetting to SUSPECT
        self.state = TrackState.CONFIRMED
        self.skip_recognition = True

        logger.info(
            f"Track {self.track_id} FAST RECOVERED → CONFIRMED ({person_id}, "
            f"cumulative={self.cumulative_confidence:.2f})"
        )

        # Reset counters
        self.inconsistent_frame_count = 0
        self.last_seen_time = time.time()
        self.start_time = time.time()

        return True

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

        # Track was seen
        self.last_seen_time = time.time()
        self.frames_since_seen = 0

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
            # If track was recovered, don't go through normal SUSPECT flow
            # The recovered state is already set and we should respect it
            if self._recovered and self.person_id is not None:
                # Already recovered, skip SUSPECT logic
                # Clear the flag so future transitions work normally
                self._recovered = False
                return False

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
                # If we were previously locked to a different person, require higher margin
                required_margin = self.t_margin
                if self.locked_person_id is not None and person_id != self.locked_person_id:
                    required_margin = self.t_margin + self.identity_switch_margin
                    logger.debug(
                        f"Track {self.track_id}: identity switch from {self.locked_person_id} to {person_id} "
                        f"requires margin {required_margin:.3f}"
                    )

                # Consistency-based margin relaxation:
                # If consistency is very high (2x required), we can trust the match even with low margin
                # This handles cases with few enrolled identities that have similar embeddings
                margin_ok = median_margin >= required_margin
                if not margin_ok and consistency >= self.consistency_count * 2:
                    # High consistency can compensate for low margin
                    # Still require some minimal margin to avoid complete ties
                    minimal_margin = 0.001
                    if median_margin >= minimal_margin:
                        margin_ok = True
                        logger.debug(
                            f"Track {self.track_id}: margin relaxed due to high consistency "
                            f"({consistency} >= {self.consistency_count * 2})"
                        )

                if (
                    median_score >= self.t_accept and
                    margin_ok and
                    consistency >= self.consistency_count
                ):
                    self.state = TrackState.FRIENDLY
                    self.person_id = person_id
                    self.confidence = median_score
                    self.inconsistent_frame_count = 0

                    # Lock this identity
                    self.locked_person_id = person_id
                    self.locked_confidence = median_score

                    # Initialize cumulative confidence if not already set (e.g., by recovery)
                    # Start at 0% for clean 0-100% progression
                    if self.cumulative_confidence < 0.01:
                        self.cumulative_confidence = 0.0  # Start at 0%

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
            # Continue comparing faces and accumulate confidence
            person_id, metrics = self.aggregator.get_aggregated_decision(
                consistency_count=self.consistency_count
            )

            if person_id is not None and metrics:
                median_score = metrics.get("median_score", 0.0)
                median_margin = metrics.get("median_margin", 0.0)
                consistency = metrics.get("consistency", 0)

                # Check if detection is still consistent with same person
                # Apply same margin relaxation as for initial acceptance
                margin_ok = median_margin >= self.t_margin
                if not margin_ok and consistency >= self.consistency_count * 2:
                    if median_margin >= 0.001:
                        margin_ok = True

                if (
                    person_id == self.person_id and
                    median_score >= self.t_accept and
                    margin_ok and
                    consistency >= self.consistency_count
                ):
                    # Continuous positive detection - ACCUMULATE CONFIDENCE
                    # Use EMA to smoothly increase cumulative confidence
                    self.confidence = max(self.confidence, median_score)

                    # Gain cumulative confidence with each positive frame
                    # Formula: cumulative = cumulative + gain_rate * (1 - cumulative)
                    # This asymptotically approaches 1.0
                    confidence_boost = self.confidence_gain_rate * (1.0 - self.cumulative_confidence)
                    self.cumulative_confidence = min(1.0, self.cumulative_confidence + confidence_boost)

                    self.inconsistent_frame_count = 0

                    logger.debug(
                        f"Track {self.track_id} FRIENDLY: {self.person_id}, "
                        f"score={self.confidence:.3f}, cumulative={self.cumulative_confidence:.3f}"
                    )

                    # Check for transition to CONFIRMED state
                    if self.cumulative_confidence >= self.confirm_threshold:
                        self.state = TrackState.CONFIRMED
                        self.skip_recognition = True

                        logger.info(
                            f"Track {self.track_id} → CONFIRMED ({self.person_id}, "
                            f"cumulative={self.cumulative_confidence:.3f}). "
                            f"Switching to tracking-only mode."
                        )
                        return True
                else:
                    # Detection inconsistent - decay confidence slightly
                    self.inconsistent_frame_count += 1
                    self.cumulative_confidence = max(
                        0.0,
                        self.cumulative_confidence - self.confidence_decay_rate
                    )

                    # For recovered tracks, be more lenient - allow same person with lower score
                    if self._recovered and person_id == self.person_id:
                        # Same person, just lower quality - don't count as inconsistent
                        self.inconsistent_frame_count = max(0, self.inconsistent_frame_count - 1)
                        logger.debug(
                            f"Track {self.track_id} FRIENDLY (recovered, lenient): {person_id}, "
                            f"score={median_score:.3f}, cumulative={self.cumulative_confidence:.3f}"
                        )
                    else:
                        logger.debug(
                            f"Track {self.track_id} FRIENDLY inconsistent frame "
                            f"({self.inconsistent_frame_count}/{self.confidence_decay_threshold}): "
                            f"detected={person_id}, expected={self.person_id}, "
                            f"cumulative={self.cumulative_confidence:.3f}"
                        )

                    # Check if we should degrade back to SUSPECT
                    if self.inconsistent_frame_count >= self.confidence_decay_threshold:
                        old_person = self.person_id
                        old_confidence = self.confidence

                        self.state = TrackState.SUSPECT
                        self.person_id = None
                        self.confidence = 0.0
                        self.cumulative_confidence = 0.0
                        self.inconsistent_frame_count = 0
                        self._recovered = False  # Clear recovery flag
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
                self.cumulative_confidence = max(
                    0.0,
                    self.cumulative_confidence - self.confidence_decay_rate
                )

                if self.inconsistent_frame_count >= self.confidence_decay_threshold:
                    old_person = self.person_id
                    old_confidence = self.confidence

                    self.state = TrackState.SUSPECT
                    self.person_id = None
                    self.confidence = 0.0
                    self.cumulative_confidence = 0.0
                    self.inconsistent_frame_count = 0
                    self._recovered = False  # Clear recovery flag
                    self.start_time = time.time()
                    self.aggregator.clear()

                    logger.info(
                        f"Track {self.track_id} FRIENDLY → SUSPECT (no detection: "
                        f"was {old_person} conf={old_confidence:.3f})"
                    )
                    return True

        elif self.state == TrackState.CONFIRMED:
            # CONFIRMED state: High-confidence tracking-only mode
            # We don't do recognition, just track the face
            # If track is lost for too long, drop back to FRIENDLY

            time_since_seen = time.time() - self.last_seen_time

            if time_since_seen > self.track_lost_timeout:
                # Track lost for too long - drop back to FRIENDLY to re-verify
                self.state = TrackState.FRIENDLY
                self.skip_recognition = False
                self.cumulative_confidence = self.confirm_threshold * 0.8  # Keep some confidence

                logger.info(
                    f"Track {self.track_id} CONFIRMED → FRIENDLY (track lost for "
                    f"{time_since_seen:.1f}s > {self.track_lost_timeout:.1f}s). "
                    f"Re-enabling recognition."
                )
                return True
            else:
                # Track is being tracked - stay in CONFIRMED
                logger.debug(
                    f"Track {self.track_id} CONFIRMED: {self.person_id}, "
                    f"cumulative={self.cumulative_confidence:.3f}, "
                    f"time_since_seen={time_since_seen:.2f}s"
                )

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
            "cumulative_confidence": self.cumulative_confidence,
            "modalities_used": self.modalities_used.copy()
        }
        return event

    def is_terminal(self) -> bool:
        """Check if state is terminal (no more processing needed).

        CONFIRMED: Only needs tracking, no recognition
        FRIENDLY: Continue recognition to build confidence
        SUSPECT: Continue recognition
        ENEMY: Continue to allow re-identification

        Returns:
            True if recognition should stop (CONFIRMED state)
        """
        return self.state == TrackState.CONFIRMED

