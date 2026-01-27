"""Unified scorer combining FAISS similarity search with optional PLDA verification.

Provides a clean interface for the recognition pipeline that:
1. Uses FAISS for fast candidate retrieval (Stage 1)
2. Optionally refines with PLDA log-likelihood ratio scoring (Stage 2)
3. Falls back gracefully when PLDA is not available
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from watchbird.index.faiss_wrapper import FaissIndex
from watchbird.index.meta_store import MetaStore
from watchbird.fusion.plda_scorer import PLDAScorer

logger = logging.getLogger(__name__)


class UnifiedScorer:
    """Unified scoring backend combining FAISS + optional PLDA.

    Stage 1 (FAISS): Fast approximate nearest neighbor search using cosine similarity.
                     Returns top-K candidates.

    Stage 2 (PLDA):  Optional probabilistic verification using log-likelihood ratios.
                     Provides better open-set rejection and calibrated scores.

    Decision Logic:
        - If PLDA enabled and model exists: Use PLDA scores + margin logic
        - Else: Fall back to existing cosine similarity thresholds
    """

    def __init__(
        self,
        faiss_index: FaissIndex,
        meta_store: MetaStore,
        plda_model_path: Optional[str] = None,
        plda_enabled: bool = False,
        faiss_k: int = 5,
        plda_llr_threshold: float = 0.0,
        plda_margin_threshold: float = 0.5,
        plda_calibrate: bool = True
    ):
        """Initialize unified scorer.

        Args:
            faiss_index: FAISS index for similarity search
            meta_store: Metadata store for person ID lookup
            plda_model_path: Path to PLDA model file (optional)
            plda_enabled: Whether to use PLDA scoring if available
            faiss_k: Number of candidates to retrieve from FAISS
            plda_llr_threshold: Minimum LLR for PLDA acceptance
            plda_margin_threshold: Minimum LLR margin for PLDA
            plda_calibrate: Whether to calibrate PLDA scores to [0, 1]
        """
        self.faiss_index = faiss_index
        self.meta_store = meta_store
        self.faiss_k = faiss_k

        # PLDA configuration
        self.plda_enabled = plda_enabled
        self.plda_model_path = plda_model_path
        self.plda_llr_threshold = plda_llr_threshold
        self.plda_margin_threshold = plda_margin_threshold
        self.plda_calibrate = plda_calibrate

        # PLDA scorer instance
        self.plda_scorer: Optional[PLDAScorer] = None

        # Try to load PLDA model if enabled
        if plda_enabled and plda_model_path:
            self._load_plda_model(plda_model_path)

    def _load_plda_model(self, path: str) -> bool:
        """Load PLDA model from file.

        Args:
            path: Path to PLDA model file

        Returns:
            True if loaded successfully
        """
        if not Path(path).exists():
            logger.info(f"PLDA model not found at {path}, will use FAISS-only scoring")
            return False

        try:
            self.plda_scorer = PLDAScorer()
            if self.plda_scorer.load(path):
                logger.info(f"PLDA scorer loaded: {self.plda_scorer}")
                return True
            else:
                self.plda_scorer = None
                return False
        except Exception as e:
            logger.error(f"Failed to load PLDA model: {e}")
            self.plda_scorer = None
            return False

    @property
    def has_plda(self) -> bool:
        """Check if PLDA scoring is available."""
        return (
            self.plda_enabled and
            self.plda_scorer is not None and
            self.plda_scorer.is_trained
        )

    def score(
        self,
        embedding: np.ndarray
    ) -> Tuple[Optional[str], float, float, Dict[str, float]]:
        """Score a probe embedding against enrolled identities.

        Args:
            embedding: Probe face embedding [embedding_dim]

        Returns:
            Tuple of (best_person_id, best_score, margin, all_scores)
            - best_person_id: Top matching identity or None
            - best_score: Score for best match (cosine sim or calibrated PLDA)
            - margin: Score difference between top two candidates
            - all_scores: Dict of person_id -> score for all candidates
        """
        # Ensure embedding is 1D
        if embedding.ndim == 2:
            embedding = embedding.squeeze(0)

        # Stage 1: FAISS candidate retrieval
        similarities, indices = self.faiss_index.search(embedding, k=self.faiss_k)

        if len(similarities) == 0 or len(indices) == 0:
            return None, 0.0, 0.0, {}

        # Flatten if needed
        similarities = similarities.flatten()
        indices = indices.flatten()

        # Map indices to person IDs
        candidates: Dict[str, List[float]] = {}
        for sim, idx in zip(similarities, indices):
            person_id = self.meta_store.get_person_id(int(idx))
            if person_id is not None:
                if person_id not in candidates:
                    candidates[person_id] = []
                candidates[person_id].append(float(sim))

        if not candidates:
            return None, 0.0, 0.0, {}

        # Get unique candidate IDs
        candidate_ids = list(candidates.keys())

        # Stage 2: Scoring
        if self.has_plda:
            return self._score_with_plda(embedding, candidate_ids, candidates)
        else:
            return self._score_faiss_only(candidates)

    def _score_faiss_only(
        self,
        candidates: Dict[str, List[float]]
    ) -> Tuple[Optional[str], float, float, Dict[str, float]]:
        """Score using FAISS cosine similarities only.

        Args:
            candidates: Dict mapping person_id to list of cosine similarities

        Returns:
            Tuple of (best_person_id, best_score, margin, all_scores)
        """
        # Aggregate by taking max similarity per person
        all_scores = {
            person_id: max(sims)
            for person_id, sims in candidates.items()
        }

        # Sort by score descending
        sorted_persons = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        best_id = sorted_persons[0][0]
        best_score = sorted_persons[0][1]
        second_score = sorted_persons[1][1] if len(sorted_persons) > 1 else 0.0
        margin = best_score - second_score

        return best_id, best_score, margin, all_scores

    def _score_with_plda(
        self,
        embedding: np.ndarray,
        candidate_ids: List[str],
        faiss_candidates: Dict[str, List[float]]
    ) -> Tuple[Optional[str], float, float, Dict[str, float]]:
        """Score using PLDA on FAISS candidates.

        Args:
            embedding: Probe embedding
            candidate_ids: List of candidate person IDs from FAISS
            faiss_candidates: FAISS similarity scores per candidate

        Returns:
            Tuple of (best_person_id, best_score, margin, all_scores)
        """
        # Get FAISS scores for validation
        faiss_scores = {
            person_id: max(sims)
            for person_id, sims in faiss_candidates.items()
        }

        # Get best FAISS result for reference
        faiss_sorted = sorted(faiss_scores.items(), key=lambda x: x[1], reverse=True)
        best_faiss_id = faiss_sorted[0][0]
        best_faiss_score = faiss_sorted[0][1]

        # If FAISS score is too low, don't trust PLDA either
        # This catches partial occlusions and poor quality embeddings
        faiss_min_threshold = 0.5  # Minimum cosine similarity to proceed with PLDA
        if best_faiss_score < faiss_min_threshold:
            logger.debug(
                f"PLDA skipped: FAISS score {best_faiss_score:.3f} < {faiss_min_threshold:.3f}"
            )
            return self._score_faiss_only(faiss_candidates)

        # Get PLDA scores for all candidates
        plda_scores = self.plda_scorer.score_batch(embedding, candidate_ids)

        # Calibrate scores to [0, 1] range if enabled
        if self.plda_calibrate:
            all_scores = {
                person_id: self.plda_scorer.calibrate_score(llr)
                for person_id, llr in plda_scores.items()
            }
        else:
            all_scores = plda_scores

        if not all_scores:
            return None, 0.0, 0.0, {}

        # Sort by calibrated score descending
        sorted_persons = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        best_plda_id, best_plda_score = sorted_persons[0]
        second_score = sorted_persons[1][1] if len(sorted_persons) > 1 else 0.0
        margin = best_plda_score - second_score

        # Get raw LLR for threshold check
        best_llr = plda_scores[best_plda_id]

        # Apply PLDA-specific thresholds
        if best_llr < self.plda_llr_threshold:
            logger.debug(
                f"PLDA reject: LLR {best_llr:.2f} < threshold {self.plda_llr_threshold:.2f}"
            )
            # Fall back to FAISS scores for this frame
            return self._score_faiss_only(faiss_candidates)

        # CRITICAL: PLDA must agree with FAISS on the winner
        # If PLDA disagrees with FAISS, trust FAISS (it's more reliable with limited training data)
        if best_plda_id != best_faiss_id:
            # PLDA and FAISS disagree - check if FAISS has a clear winner
            faiss_margin = best_faiss_score - (faiss_sorted[1][1] if len(faiss_sorted) > 1 else 0.0)

            if faiss_margin > 0.05:  # FAISS has a clear winner
                logger.debug(
                    f"PLDA disagrees with FAISS: PLDA={best_plda_id}, FAISS={best_faiss_id}. "
                    f"Using FAISS (margin={faiss_margin:.3f})"
                )
                return self._score_faiss_only(faiss_candidates)
            else:
                # Both are uncertain, be conservative
                logger.debug(
                    f"PLDA/FAISS disagree and both uncertain. Using FAISS."
                )
                return self._score_faiss_only(faiss_candidates)

        # PLDA and FAISS agree - use PLDA score (provides better calibration)
        return best_plda_id, best_plda_score, margin, all_scores


    def get_scoring_info(self) -> Dict[str, any]:
        """Get information about current scoring configuration.

        Returns:
            Dict with scoring backend info
        """
        info: Dict[str, any] = {
            'backend': 'plda+faiss' if self.has_plda else 'faiss',
            'faiss_vectors': self.faiss_index.num_vectors,
            'faiss_k': self.faiss_k,
            'plda_enabled': self.plda_enabled,
            'plda_available': self.has_plda,
        }

        if self.has_plda:
            info['plda_identities'] = len(self.plda_scorer.identity_models)
            info['plda_llr_threshold'] = self.plda_llr_threshold
            info['plda_margin_threshold'] = self.plda_margin_threshold

        return info


def create_scorer_from_config(
    faiss_index: FaissIndex,
    meta_store: MetaStore,
    config
) -> UnifiedScorer:
    """Create UnifiedScorer from configuration.

    Args:
        faiss_index: FAISS index instance
        meta_store: Metadata store instance
        config: Configuration object

    Returns:
        Configured UnifiedScorer instance
    """
    # Get PLDA configuration
    plda_config = config.get('plda', {}) if hasattr(config, 'get') else {}

    if isinstance(config, dict):
        plda_enabled = plda_config.get('enabled', False)
        plda_model_path = plda_config.get('model_path', 'data/index/plda.npz')
        faiss_k = plda_config.get('faiss_k', 5)
        llr_threshold = plda_config.get('llr_threshold', 0.0)
        margin_threshold = plda_config.get('margin_threshold', 0.5)
        calibrate = plda_config.get('calibrate', True)
    else:
        # Config object with get() method
        plda_enabled = config.get('plda.enabled', False)
        plda_model_path = config.get('plda.model_path', 'data/index/plda.npz')
        faiss_k = config.get('plda.faiss_k', 5)
        llr_threshold = config.get('plda.llr_threshold', 0.0)
        margin_threshold = config.get('plda.margin_threshold', 0.5)
        calibrate = config.get('plda.calibrate', True)

    return UnifiedScorer(
        faiss_index=faiss_index,
        meta_store=meta_store,
        plda_model_path=plda_model_path,
        plda_enabled=plda_enabled,
        faiss_k=faiss_k,
        plda_llr_threshold=llr_threshold,
        plda_margin_threshold=margin_threshold,
        plda_calibrate=calibrate
    )
