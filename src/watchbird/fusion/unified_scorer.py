"""Unified scorer combining FAISS similarity search with PLDA likelihood ratio scoring.

Provides a clean interface for the recognition pipeline that:
1. Uses FAISS for fast candidate retrieval (Stage 1)
2. Refines with PLDA log-likelihood ratio scoring (Stage 2)
3. Uses LLR thresholds for robust unknown/impostor rejection

The key insight is that PLDA LLR scoring naturally handles the "unknown" problem:
- LLR > 0: More likely same person
- LLR < 0: More likely different person (potential unknown)
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
    """Unified scoring backend combining FAISS + PLDA likelihood ratio scoring.

    Stage 1 (FAISS): Fast approximate nearest neighbor search using cosine similarity.
                     Returns top-K candidates for PLDA scoring.

    Stage 2 (PLDA):  Probabilistic verification using log-likelihood ratios.
                     LLR > 0 indicates same person is more likely than different.
                     LLR < 0 indicates different person (or unknown) is more likely.
                     This naturally handles the "unknown" rejection problem.

    Decision Logic:
        - If PLDA enabled and model exists: Use PLDA LLR scores
        - LLR threshold determines unknown rejection
        - Margin threshold ensures confident discrimination
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
        """Score using PLDA log-likelihood ratios on FAISS candidates.

        Uses the new PLDA LLR approach for unknown rejection:
        - LLR > 0: More likely same person
        - LLR < 0: More likely different/unknown

        Args:
            embedding: Probe embedding
            candidate_ids: List of candidate person IDs from FAISS
            faiss_candidates: FAISS similarity scores per candidate

        Returns:
            Tuple of (best_person_id, best_score, margin, all_scores)
        """
        # Get FAISS scores for reference
        faiss_scores = {
            person_id: max(sims)
            for person_id, sims in faiss_candidates.items()
        }

        # Get best FAISS result for validation
        faiss_sorted = sorted(faiss_scores.items(), key=lambda x: x[1], reverse=True)
        best_faiss_id = faiss_sorted[0][0]
        best_faiss_score = faiss_sorted[0][1]

        # If FAISS score is too low, embeddings may be too noisy for PLDA
        faiss_min_threshold = 0.45
        if best_faiss_score < faiss_min_threshold:
            logger.debug(
                f"PLDA skipped: FAISS score {best_faiss_score:.3f} < {faiss_min_threshold:.3f}"
            )
            return self._score_faiss_only(faiss_candidates)

        # Get PLDA LLR scores for all candidates
        plda_scores = self.plda_scorer.score_batch(embedding, candidate_ids)

        # Sort by LLR (higher = more likely same person)
        sorted_plda = sorted(plda_scores.items(), key=lambda x: x[1], reverse=True)

        if not sorted_plda:
            return None, 0.0, 0.0, {}

        best_plda_id, best_llr = sorted_plda[0]
        second_llr = sorted_plda[1][1] if len(sorted_plda) > 1 else float('-inf')
        plda_margin = best_llr - second_llr

        # KEY INSIGHT: Use LLR directly for unknown rejection
        # LLR < threshold means "unknown" is more likely than any enrolled identity
        if best_llr < self.plda_llr_threshold:
            logger.debug(
                f"PLDA reject (unknown): LLR {best_llr:.2f} < threshold {self.plda_llr_threshold:.2f}"
            )
            # Return None to indicate unknown, but include the LLR for debugging
            return None, best_llr, plda_margin, plda_scores

        # Check margin - if top two candidates are too close, not confident
        if plda_margin < self.plda_margin_threshold and len(sorted_plda) > 1:
            logger.debug(
                f"PLDA uncertain: margin {plda_margin:.2f} < threshold {self.plda_margin_threshold:.2f}"
            )
            # Still return the best match, but the low margin will be considered by caller
            pass

        # PLDA and FAISS agreement check
        # If they disagree and FAISS has high confidence, investigate
        if best_plda_id != best_faiss_id:
            faiss_margin = best_faiss_score - (faiss_sorted[1][1] if len(faiss_sorted) > 1 else 0.0)

            # If FAISS is very confident but PLDA disagrees, there may be an issue
            if faiss_margin > 0.15 and best_faiss_score > 0.7:
                logger.debug(
                    f"PLDA/FAISS disagree: PLDA={best_plda_id}(LLR={best_llr:.2f}), "
                    f"FAISS={best_faiss_id}(sim={best_faiss_score:.3f}). "
                    f"Using PLDA decision."
                )

        # Calibrate scores to [0, 1] range if enabled
        if self.plda_calibrate:
            all_scores = {
                person_id: self.plda_scorer.calibrate_score(llr)
                for person_id, llr in plda_scores.items()
            }
            best_score = self.plda_scorer.calibrate_score(best_llr)
            calibrated_margin = best_score - (
                self.plda_scorer.calibrate_score(second_llr)
                if np.isfinite(second_llr) else 0.0
            )
        else:
            all_scores = plda_scores
            best_score = best_llr
            calibrated_margin = plda_margin

        logger.debug(
            f"PLDA result: id={best_plda_id}, LLR={best_llr:.2f}, "
            f"calibrated={best_score:.3f}, margin={calibrated_margin:.3f}"
        )

        return best_plda_id, best_score, calibrated_margin, all_scores


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
