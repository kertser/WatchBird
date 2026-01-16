"""Metadata store for FAISS index."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MetaStore:
    """Metadata store for embeddings in FAISS index."""

    def __init__(self, meta_path: str):
        """Initialize metadata store.

        Args:
            meta_path: Path to metadata JSONL file
        """
        self.meta_path = Path(meta_path)
        self.metadata: List[Dict] = []

    def add(
        self,
        vec_id: int,
        person_id: str,
        modality: str,
        source: str,
        quality: float
    ) -> None:
        """Add metadata entry.

        Args:
            vec_id: Vector ID in FAISS index
            person_id: Person identifier
            modality: Modality type (face, reid, gait)
            source: Source file path
            quality: Quality score
        """
        entry = {
            "vec_id": vec_id,
            "person_id": person_id,
            "modality": modality,
            "source": source,
            "quality": quality,
            "ts": datetime.utcnow().isoformat() + "Z"
        }
        self.metadata.append(entry)

    def get(self, vec_id: int) -> Optional[Dict]:
        """Get metadata by vector ID.

        Args:
            vec_id: Vector ID

        Returns:
            Metadata dict or None if not found
        """
        for entry in self.metadata:
            if entry["vec_id"] == vec_id:
                return entry
        return None

    def get_person_id(self, vec_id: int) -> Optional[str]:
        """Get person ID by vector ID.

        Args:
            vec_id: Vector ID

        Returns:
            Person ID or None if not found
        """
        entry = self.get(vec_id)
        return entry["person_id"] if entry else None

    def save(self) -> bool:
        """Save metadata to JSONL file.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.meta_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.meta_path, "w") as f:
                for entry in self.metadata:
                    f.write(json.dumps(entry) + "\n")

            logger.info(f"Saved {len(self.metadata)} metadata entries to {self.meta_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return False

    def load(self) -> bool:
        """Load metadata from JSONL file.

        Returns:
            True if successful, False otherwise
        """
        if not self.meta_path.exists():
            logger.warning(f"Metadata file not found: {self.meta_path}")
            return False

        try:
            self.metadata = []

            with open(self.meta_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self.metadata.append(entry)

            logger.info(f"Loaded {len(self.metadata)} metadata entries from {self.meta_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return False

