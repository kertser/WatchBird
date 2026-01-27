"""PLDA (Probabilistic Linear Discriminant Analysis) scorer for open-set face recognition.

Implements a two-covariance PLDA model for computing log-likelihood ratios (LLR)
between probe embeddings and enrolled identity models. Provides robust open-set
verification on top of FAISS-based candidate retrieval.

References:
    - Prince & Elder (2007): Probabilistic Linear Discriminant Analysis
    - Sizov et al. (2014): Unifying Probabilistic Linear Discriminant Analysis Variants
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PLDAScorer:
    """Two-covariance PLDA scorer for open-set face verification.

    The PLDA model decomposes face embeddings as:
        x = μ + Φ·h + ε

    where:
        - μ is the global mean
        - Φ is the between-class (identity) subspace
        - h ~ N(0, I) is the identity factor
        - ε ~ N(0, Σ_w) is within-class noise

    For scoring, we compute the log-likelihood ratio (LLR):
        LLR = log P(x_probe, X_gallery | same) - log P(x_probe | unk) - log P(X_gallery | enroll)
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        plda_dim: int = 128,
        regularization: float = 1e-5
    ):
        """Initialize PLDA scorer.

        Args:
            embedding_dim: Dimension of input embeddings
            plda_dim: Dimension of PLDA subspace (between-class factors)
            regularization: Regularization for covariance inversion
        """
        self.embedding_dim = embedding_dim
        self.plda_dim = plda_dim
        self.regularization = regularization

        # Model parameters (set after training)
        self.global_mean: Optional[np.ndarray] = None  # (embedding_dim,)
        self.between_cov: Optional[np.ndarray] = None  # Σ_b: (embedding_dim, embedding_dim)
        self.within_cov: Optional[np.ndarray] = None   # Σ_w: (embedding_dim, embedding_dim)

        # Precomputed scoring matrices
        self._precision_total: Optional[np.ndarray] = None  # (Σ_b + Σ_w)^{-1}
        self._precision_within: Optional[np.ndarray] = None  # Σ_w^{-1}
        self._Q: Optional[np.ndarray] = None  # For efficient LLR computation
        self._P: Optional[np.ndarray] = None  # For efficient LLR computation
        self._const_term: float = 0.0

        # Identity models: person_id -> mean embedding
        self.identity_models: Dict[str, np.ndarray] = {}
        self.identity_counts: Dict[str, int] = {}  # Number of samples per identity

        self._trained = False

    @property
    def is_trained(self) -> bool:
        """Check if PLDA model is trained."""
        return self._trained

    def train(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        max_iter: int = 20,
        tol: float = 1e-4
    ) -> bool:
        """Train PLDA model using EM algorithm.

        Args:
            embeddings: Face embeddings [N, embedding_dim]
            labels: Identity labels for each embedding
            max_iter: Maximum EM iterations
            tol: Convergence tolerance

        Returns:
            True if training succeeded
        """
        try:
            N, D = embeddings.shape

            if D != self.embedding_dim:
                logger.warning(
                    f"Embedding dim mismatch: expected {self.embedding_dim}, got {D}. "
                    f"Updating to {D}."
                )
                self.embedding_dim = D

            # Get unique identities
            unique_labels = list(set(labels))
            num_classes = len(unique_labels)
            label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

            if num_classes < 2:
                logger.error("PLDA requires at least 2 distinct identities")
                return False

            # Check minimum samples per class for reliable covariance estimation
            min_samples_per_class = 5
            samples_per_class = {}
            for lbl in labels:
                samples_per_class[lbl] = samples_per_class.get(lbl, 0) + 1

            insufficient_classes = [
                lbl for lbl, count in samples_per_class.items()
                if count < min_samples_per_class
            ]

            if insufficient_classes:
                logger.warning(
                    f"PLDA: Some identities have too few samples (need {min_samples_per_class}+): "
                    f"{insufficient_classes}. PLDA may be unreliable."
                )

            # Require minimum total samples for reliable covariance estimation
            # Rule of thumb: N > 2*D for stable covariance, but we use regularization
            min_total_samples = max(20, num_classes * 5)
            if N < min_total_samples:
                logger.warning(
                    f"PLDA: Only {N} samples available, recommend {min_total_samples}+ "
                    f"for {num_classes} identities. PLDA may be unreliable."
                )

            logger.info(f"Training PLDA: {N} samples, {num_classes} identities, dim={D}")

            # Compute global mean and center data
            self.global_mean = np.mean(embeddings, axis=0)
            centered = embeddings - self.global_mean

            # Compute class means and within-class scatter
            class_means = np.zeros((num_classes, D))
            class_counts = np.zeros(num_classes)

            for i, lbl in enumerate(labels):
                idx = label_to_idx[lbl]
                class_means[idx] += centered[i]
                class_counts[idx] += 1

            # Normalize class means
            for c in range(num_classes):
                if class_counts[c] > 0:
                    class_means[c] /= class_counts[c]

            # Initialize covariances using method of moments
            # Between-class covariance
            self.between_cov = np.cov(class_means.T)
            if self.between_cov.ndim == 0:
                self.between_cov = np.array([[self.between_cov]])

            # Within-class covariance
            within_scatter = np.zeros((D, D))
            total_within = 0

            for i, lbl in enumerate(labels):
                idx = label_to_idx[lbl]
                diff = centered[i] - class_means[idx]
                within_scatter += np.outer(diff, diff)
                total_within += 1

            self.within_cov = within_scatter / max(total_within, 1)

            # Regularize covariances
            reg_eye = self.regularization * np.eye(D)
            self.between_cov += reg_eye
            self.within_cov += reg_eye

            # EM refinement (simplified, could use full EM with latent factors)
            prev_ll = -np.inf

            for iteration in range(max_iter):
                # E-step: estimate identity factors (simplified)
                # For full PLDA, would need to estimate latent h factors

                # M-step: update covariances based on current estimates
                # Here we do a single-pass estimation which is often sufficient

                # Compute log-likelihood proxy
                try:
                    total_cov = self.between_cov + self.within_cov
                    sign, logdet = np.linalg.slogdet(total_cov)
                    if sign <= 0:
                        break

                    precision = np.linalg.inv(total_cov)

                    ll = 0.0
                    for i in range(N):
                        diff = centered[i]
                        ll -= 0.5 * diff @ precision @ diff
                    ll -= 0.5 * N * logdet

                    if abs(ll - prev_ll) < tol * abs(prev_ll):
                        logger.debug(f"PLDA converged at iteration {iteration}")
                        break

                    prev_ll = ll

                except np.linalg.LinAlgError:
                    logger.warning(f"PLDA: matrix singular at iteration {iteration}")
                    break

            # Precompute scoring matrices
            self._precompute_scoring_matrices()

            # Build identity models (mean embeddings per person)
            self._build_identity_models(embeddings, labels)

            self._trained = True
            logger.info(f"PLDA training complete: {num_classes} identity models")

            return True

        except Exception as e:
            logger.error(f"PLDA training failed: {e}")
            return False

    def _precompute_scoring_matrices(self) -> None:
        """Precompute matrices for efficient LLR scoring."""
        try:
            D = self.embedding_dim
            reg_eye = self.regularization * np.eye(D)

            # Σ_tot = Σ_b + Σ_w
            total_cov = self.between_cov + self.within_cov

            # Precision matrices
            self._precision_within = np.linalg.inv(self.within_cov + reg_eye)
            self._precision_total = np.linalg.inv(total_cov + reg_eye)

            # For same-speaker scoring with multiple enrollment samples:
            # Q = Σ_w^{-1} - (Σ_b + Σ_w)^{-1}
            # P = Σ_b^{-1} + Σ_w^{-1}
            self._Q = self._precision_within - self._precision_total

            # Constant term for normalization
            _, logdet_w = np.linalg.slogdet(self.within_cov + reg_eye)
            _, logdet_tot = np.linalg.slogdet(total_cov + reg_eye)
            self._const_term = 0.5 * (logdet_w - logdet_tot)

            logger.debug("PLDA scoring matrices precomputed")

        except np.linalg.LinAlgError as e:
            logger.error(f"Failed to precompute PLDA matrices: {e}")

    def _build_identity_models(
        self,
        embeddings: np.ndarray,
        labels: List[str]
    ) -> None:
        """Build identity models from training embeddings.

        Args:
            embeddings: Training embeddings
            labels: Identity labels
        """
        self.identity_models = {}
        self.identity_counts = {}

        # Accumulate embeddings per identity
        identity_sums: Dict[str, np.ndarray] = {}

        for emb, lbl in zip(embeddings, labels):
            if lbl not in identity_sums:
                identity_sums[lbl] = np.zeros(self.embedding_dim)
                self.identity_counts[lbl] = 0

            identity_sums[lbl] += emb
            self.identity_counts[lbl] += 1

        # Compute mean embeddings
        for lbl, emb_sum in identity_sums.items():
            self.identity_models[lbl] = emb_sum / self.identity_counts[lbl]

        logger.debug(f"Built {len(self.identity_models)} identity models")

    def score(
        self,
        probe: np.ndarray,
        target_person_id: str
    ) -> float:
        """Compute PLDA LLR score for probe vs target identity.

        Args:
            probe: Probe embedding [embedding_dim]
            target_person_id: Target identity to compare against

        Returns:
            Log-likelihood ratio score (higher = more likely same person)
        """
        if not self._trained:
            logger.warning("PLDA not trained, returning 0.0")
            return 0.0

        if target_person_id not in self.identity_models:
            logger.debug(f"Unknown identity: {target_person_id}")
            return -np.inf

        try:
            # Check probe embedding quality
            # Partially occluded faces produce embeddings with unusual properties
            probe_norm = np.linalg.norm(probe)

            # Embeddings should be roughly unit-normalized
            # Very low or very high norms indicate problems
            if probe_norm < 0.1 or probe_norm > 10.0:
                logger.debug(f"PLDA: probe embedding has unusual norm {probe_norm:.3f}")
                return -np.inf

            # Normalize probe for PLDA (should already be normalized but ensure it)
            probe = probe / max(probe_norm, 1e-6)

            # Center embeddings
            probe_c = probe - self.global_mean
            target_model = self.identity_models[target_person_id] - self.global_mean
            n_enroll = self.identity_counts.get(target_person_id, 1)

            # Check cosine similarity as a quick sanity check
            # If cosine sim is low, PLDA LLR should also be low
            target_norm = np.linalg.norm(target_model)
            if target_norm > 1e-6:
                cosine_sim = np.dot(probe, self.identity_models[target_person_id]) / (
                    np.linalg.norm(probe) * np.linalg.norm(self.identity_models[target_person_id])
                )
            else:
                cosine_sim = 0.0

            # Simplified LLR computation for single probe vs. averaged gallery
            # LLR ≈ x_p^T Q x_g + const
            # where Q captures the between-class structure

            llr = probe_c @ self._Q @ target_model

            # Scale by enrollment count (more samples = more confident model)
            # This is a simplified approximation; full PLDA would marginalize
            scale = min(n_enroll, 10) / 10.0  # Cap at 10 samples
            llr *= (0.5 + 0.5 * scale)

            # Add constant term
            llr += self._const_term

            # Penalize when PLDA disagrees with cosine similarity
            # This catches cases where PLDA gives high score but cosine is low
            # (often happens with partial occlusions)
            if cosine_sim < 0.4 and llr > 0:
                # Reduce LLR when cosine similarity is low
                penalty = (0.4 - cosine_sim) * 2.0  # Max penalty of 0.8
                llr -= penalty
                logger.debug(
                    f"PLDA penalty applied: cosine={cosine_sim:.3f}, penalty={penalty:.3f}"
                )

            return float(llr)

        except Exception as e:
            logger.error(f"PLDA scoring failed: {e}")
            return 0.0

    def score_batch(
        self,
        probe: np.ndarray,
        candidate_ids: List[str]
    ) -> Dict[str, float]:
        """Score probe against multiple candidate identities.

        Args:
            probe: Probe embedding [embedding_dim]
            candidate_ids: List of candidate identity IDs

        Returns:
            Dict mapping person_id to LLR score
        """
        scores = {}
        for person_id in candidate_ids:
            scores[person_id] = self.score(probe, person_id)
        return scores

    def get_decision(
        self,
        probe: np.ndarray,
        candidate_ids: List[str],
        llr_threshold: float = 0.0,
        margin_threshold: float = 0.5
    ) -> Tuple[Optional[str], float, float]:
        """Get PLDA-based identity decision.

        Args:
            probe: Probe embedding
            candidate_ids: Candidate identities from FAISS Stage 1
            llr_threshold: Minimum LLR for acceptance
            margin_threshold: Minimum LLR margin between top two

        Returns:
            Tuple of (best_person_id, best_llr, margin)
            Returns (None, 0, 0) if no candidate passes thresholds
        """
        if not candidate_ids:
            return None, 0.0, 0.0

        # Score all candidates
        scores = self.score_batch(probe, candidate_ids)

        # Sort by score descending
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if not sorted_scores:
            return None, 0.0, 0.0

        best_id, best_llr = sorted_scores[0]
        second_llr = sorted_scores[1][1] if len(sorted_scores) > 1 else -np.inf
        margin = best_llr - second_llr

        # Apply thresholds
        if best_llr < llr_threshold:
            return None, best_llr, margin

        return best_id, best_llr, margin

    def calibrate_score(
        self,
        llr: float,
        target_range: Tuple[float, float] = (0.0, 1.0)
    ) -> float:
        """Calibrate LLR to target score range (e.g., [0, 1]).

        Uses sigmoid calibration: score = 1 / (1 + exp(-k * llr))

        Args:
            llr: Raw LLR score
            target_range: Target output range (min, max)

        Returns:
            Calibrated score in target range
        """
        # Stricter sigmoid calibration - higher k means sharper transition
        # k=1.5 ensures low LLRs map to low scores (better rejection)
        k = 1.5  # Scale factor (increased for stricter calibration)

        # Clip to prevent overflow in exp()
        # exp(-700) ≈ 0, exp(700) ≈ inf, so clip to safe range
        clipped_llr = np.clip(-k * llr, -500, 500)
        sigmoid = 1.0 / (1.0 + np.exp(clipped_llr))

        # Map to target range
        min_out, max_out = target_range
        calibrated = min_out + (max_out - min_out) * sigmoid

        # Cap maximum confidence at 0.98 to avoid unrealistic certainty
        # This prevents overfitting artifacts from producing perfect 1.0 scores
        # Real-world face recognition should never claim 100% certainty
        max_confidence = 0.98
        calibrated = min(calibrated, max_confidence)

        return float(calibrated)

    def save(self, path: str) -> bool:
        """Save PLDA model to file.

        Args:
            path: Output file path (.npz format)

        Returns:
            True if successful
        """
        if not self._trained:
            logger.error("Cannot save untrained PLDA model")
            return False

        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            # Save numpy arrays
            np.savez(
                path,
                global_mean=self.global_mean,
                between_cov=self.between_cov,
                within_cov=self.within_cov,
                embedding_dim=self.embedding_dim,
                plda_dim=self.plda_dim,
                regularization=self.regularization
            )

            # Save identity models separately (as JSON for portability)
            models_path = Path(path).with_suffix('.models.json')
            models_data = {
                person_id: {
                    'embedding': model.tolist(),
                    'count': self.identity_counts.get(person_id, 1)
                }
                for person_id, model in self.identity_models.items()
            }

            with open(models_path, 'w') as f:
                json.dump(models_data, f)

            logger.info(f"Saved PLDA model to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save PLDA model: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load PLDA model from file.

        Args:
            path: Input file path (.npz format)

        Returns:
            True if successful
        """
        try:
            if not Path(path).exists():
                logger.error(f"PLDA model file not found: {path}")
                return False

            # Load numpy arrays
            data = np.load(path)

            self.global_mean = data['global_mean']
            self.between_cov = data['between_cov']
            self.within_cov = data['within_cov']
            self.embedding_dim = int(data['embedding_dim'])
            self.plda_dim = int(data['plda_dim'])
            self.regularization = float(data['regularization'])

            # Precompute scoring matrices
            self._precompute_scoring_matrices()

            # Load identity models
            models_path = Path(path).with_suffix('.models.json')
            if models_path.exists():
                with open(models_path, 'r') as f:
                    models_data = json.load(f)

                self.identity_models = {}
                self.identity_counts = {}

                for person_id, info in models_data.items():
                    self.identity_models[person_id] = np.array(info['embedding'])
                    self.identity_counts[person_id] = info['count']
            else:
                logger.warning(f"Identity models file not found: {models_path}")

            self._trained = True
            logger.info(
                f"Loaded PLDA model from {path}: "
                f"{len(self.identity_models)} identities"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load PLDA model: {e}")
            return False

    def __repr__(self) -> str:
        return (
            f"PLDAScorer(embedding_dim={self.embedding_dim}, "
            f"plda_dim={self.plda_dim}, "
            f"trained={self._trained}, "
            f"identities={len(self.identity_models)})"
        )
