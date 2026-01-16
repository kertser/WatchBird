"""Event output handling."""

import json
import logging
import sys
from typing import Dict, TextIO

logger = logging.getLogger(__name__)


class EventEmitter:
    """Event emitter for runtime classification events."""

    def __init__(self, output_stream: TextIO = sys.stdout):
        """Initialize event emitter.

        Args:
            output_stream: Output stream for events (default: stdout)
        """
        self.output_stream = output_stream

    def emit(self, event: Dict) -> None:
        """Emit event as JSON line.

        Args:
            event: Event dictionary
        """
        try:
            json_line = json.dumps(event)
            self.output_stream.write(json_line + "\n")
            self.output_stream.flush()
        except Exception as e:
            logger.error(f"Failed to emit event: {e}")

    def emit_state_change(
        self,
        track_id: int,
        state: str,
        person_id: str = None,
        confidence: float = 0.0,
        modalities_used: Dict[str, float] = None
    ) -> None:
        """Emit state change event.

        Args:
            track_id: Track identifier
            state: New state (FRIENDLY, ENEMY, SUSPECT)
            person_id: Person identifier (for FRIENDLY)
            confidence: Confidence score
            modalities_used: Dict of modality reliabilities
        """
        import time

        event = {
            "ts": time.time(),
            "track_id": track_id,
            "state": state,
            "person_id": person_id,
            "confidence": confidence,
            "modalities_used": modalities_used or {}
        }

        self.emit(event)

