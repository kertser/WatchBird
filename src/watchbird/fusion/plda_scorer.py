"""Regularized PLDA (Probabilistic Linear Discriminant Analysis) scorer.

Implements a two-covariance PLDA model with proper log-likelihood ratio (LLR)
scoring for open-set face recognition. Uses regularization to handle:
- Dispersed enrollment embeddings (varied poses/quality)
- Noisy probe embeddings (partial occlusion, extreme angles)
- Unknown/impostor rejection via likelihood ratios

Key insight: LLR > 0 means "more likely same person than different"
           LLR < 0 means "more likely different/unknown"

References:
    - Prince & Elder (2007): Probabilistic Linear Discriminant Analysis
    - Sizov et al. (2014): Unifying Probabilistic Linear Discriminant Analysis Variants
    - Garcia-Romero & Espy-Wilson (2011): Analysis of i-vector Length Normalization
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)


@dataclass
class PLDAConfig:
    """Configuration for PLDA model."""

    # Dimensionality
    embedding_dim: int = 512
    latent_dim: int = 128  # Reduced dimensionality for stability

    # Regularization (KEY for noisy embeddings)
    between_class_reg: float = 0.1    # Shrinkage for between-class covariance
    within_class_reg: float = 0.3     # Shrinkage for within-class covariance (higher = more tolerant)
    min_eigenvalue: float = 1e-4      # Floor for eigenvalues to ensure positive definiteness

    # Training
    min_samples_per_class: int = 3    # Minimum samples per identity for reliable estimation
    max_em_iterations: int = 10       # EM iterations (usually converges fast)
    em_tolerance: float = 1e-4        # Convergence tolerance

    # Scoring calibration
    llr_scale: float = 1.0            # Scale factor for LLR scores
    llr_shift: float = 0.0            # Shift for LLR scores (learned during calibration)


@dataclass
class PLDAModel:
    """Trained PLDA model parameters."""

    # Core parameters
    global_mean: np.ndarray = None           # (D,) global mean
    whiten_transform: np.ndarray = None      # (D, K) whitening + dimensionality reduction
    phi_b: np.ndarray = None                 # (K, K) between-class covariance in latent space
    phi_w: np.ndarray = None                 # (K, K) within-class covariance in latent space

    # Precomputed matrices for efficient scoring
    phi_b_inv: np.ndarray = None             # (K, K)
    phi_w_inv: np.ndarray = None             # (K, K)
    lambda_mat: np.ndarray = None            # (K, K) = (Φ_b^-1 + Φ_w^-1)^-1
    gamma_mat: np.ndarray = None             # (K, K) = (Φ_b^-1 + 2*Φ_w^-1)^-1

    # Log-determinant terms for scoring
    log_det_lambda: float = 0.0
    log_det_gamma: float = 0.0
    log_det_phi_b: float = 0.0

    # Calibration parameters
    llr_mean: float = 0.0                    # Mean of impostor LLR distribution
    llr_std: float = 1.0                     # Std of LLR distribution

    # Identity models
    identity_embeddings: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    identity_centroids: Dict[str, np.ndarray] = field(default_factory=dict)
    identity_latent: Dict[str, np.ndarray] = field(default_factory=dict)  # Precomputed latent representations


class PLDAScorer:
    """Regularized PLDA scorer for open-set face verification.

    The PLDA generative model assumes:
        x = μ + Φ·h + ε

    where:
        - μ is the global mean
        - Φ is the between-class (identity) subspace matrix
        - h ~ N(0, I) is the latent identity factor
        - ε ~ N(0, Σ_w) is within-class (session) noise

    For verification, we compute the log-likelihood ratio:
        LLR = log P(x_probe, x_gallery | same identity)
            - log P(x_probe | different)
            - log P(x_gallery | different)

    LLR > 0 indicates same identity is more likely.
    LLR < 0 indicates different identity (or unknown) is more likely.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        plda_dim: int = 128,
        regularization: float = 1e-5,
        config: Optional[PLDAConfig] = None
    ):
        """Initialize PLDA scorer.

        Args:
            embedding_dim: Dimension of input embeddings
            plda_dim: Dimension of PLDA latent subspace
            regularization: Base regularization (overridden by config if provided)
            config: Full PLDA configuration. Uses defaults if None.
        """
        if config is not None:
            self.config = config
        else:
            self.config = PLDAConfig(
                embedding_dim=embedding_dim,
                latent_dim=plda_dim,
                between_class_reg=0.1,
                within_class_reg=0.3,
                min_eigenvalue=max(regularization, 1e-5)
            )

        self.model: Optional[PLDAModel] = None
        self._trained = False

    @property
    def is_trained(self) -> bool:
        """Check if PLDA model is trained."""
        return self._trained and self.model is not None

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.config.embedding_dim

    @property
    def plda_dim(self) -> int:
        """Get PLDA latent dimension."""
        return self.config.latent_dim

    @property
    def regularization(self) -> float:
        """Get minimum eigenvalue regularization."""
        return self.config.min_eigenvalue

    @property
    def identity_models(self) -> Dict[str, np.ndarray]:
        """Get identity centroid embeddings (for compatibility)."""
        if self.model is None:
            return {}
        return self.model.identity_centroids

    @property
    def identity_counts(self) -> Dict[str, int]:
        """Get count of embeddings per identity."""
        if self.model is None:
            return {}
        return {
            pid: len(embs)
            for pid, embs in self.model.identity_embeddings.items()
        }

    def train(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        max_iter: int = 20,
        tol: float = 1e-4
    ) -> bool:
        """Train regularized PLDA model.

        Uses method-of-moments estimation with regularization for robustness
        with limited training data and dispersed embeddings.

        Args:
            embeddings: Face embeddings [N, D]
            labels: Identity labels for each embedding
            max_iter: Maximum EM iterations (kept for compatibility)
            tol: Convergence tolerance (kept for compatibility)

        Returns:
            True if training succeeded
        """
        try:
            embeddings = embeddings.astype(np.float64)
            N, D = embeddings.shape

            # Update config if dimension differs
            if D != self.config.embedding_dim:
                logger.info(f"Updating embedding_dim from {self.config.embedding_dim} to {D}")
                self.config.embedding_dim = D

            # Get unique identities and validate
            unique_labels = list(set(labels))
            num_classes = len(unique_labels)
            label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

            if num_classes < 2:
                logger.error("PLDA requires at least 2 distinct identities")
                return False

            logger.info(
                f"Training PLDA: {N} samples, {num_classes} identities, "
                f"dim={D}, latent_dim={self.config.latent_dim}"
            )

            # Validate samples per class
            samples_per_class = {}
            for lbl in labels:
                samples_per_class[lbl] = samples_per_class.get(lbl, 0) + 1

            insufficient = [
                lbl for lbl, cnt in samples_per_class.items()
                if cnt < self.config.min_samples_per_class
            ]
            if insufficient:
                logger.warning(
                    f"Some identities have few samples (<{self.config.min_samples_per_class}): "
                    f"{insufficient}. Using heavy regularization."
                )

            # Step 1: Compute global mean and center data
            global_mean = embeddings.mean(axis=0)
            X = embeddings - global_mean

            # Step 2: Compute class statistics
            class_means = np.zeros((num_classes, D))
            class_counts = np.zeros(num_classes)

            for i, lbl in enumerate(labels):
                idx = label_to_idx[lbl]
                class_means[idx] += X[i]
                class_counts[idx] += 1

            # Normalize class means
            for c in range(num_classes):
                if class_counts[c] > 0:
                    class_means[c] /= class_counts[c]

            # Step 3: Compute scatter matrices with regularization
            # Between-class scatter: S_b = Σ_c (μ_c - μ)(μ_c - μ)^T
            between_scatter = (class_means.T @ class_means) / num_classes

            # Within-class scatter: S_w = Σ_i (x_i - μ_{y_i})(x_i - μ_{y_i})^T
            within_scatter = np.zeros((D, D))
            for i, lbl in enumerate(labels):
                idx = label_to_idx[lbl]
                diff = X[i] - class_means[idx]
                within_scatter += np.outer(diff, diff)
            within_scatter /= N

            # Step 4: Apply regularization (shrinkage toward identity)
            # This is KEY for handling dispersed/noisy embeddings
            between_trace = np.trace(between_scatter) / D
            within_trace = np.trace(within_scatter) / D

            reg_b = self.config.between_class_reg
            reg_w = self.config.within_class_reg

            S_b = (1 - reg_b) * between_scatter + reg_b * between_trace * np.eye(D)
            S_w = (1 - reg_w) * within_scatter + reg_w * within_trace * np.eye(D)

            # Step 5: Compute whitening transform via generalized eigendecomposition
            # We want to simultaneously diagonalize S_b and S_w
            # Solve: S_b @ v = λ * S_w @ v
            try:
                eigvals, eigvecs = linalg.eigh(S_b, S_w)
            except linalg.LinAlgError:
                # Fallback: whiten with S_w only
                logger.warning("Generalized eigendecomposition failed, using S_w whitening")
                eigvals_w, eigvecs_w = linalg.eigh(S_w)
                eigvals_w = np.maximum(eigvals_w, 1e-6)
                whiten = eigvecs_w @ np.diag(1.0 / np.sqrt(eigvals_w))

                # Project S_b to whitened space
                S_b_white = whiten.T @ S_b @ whiten
                eigvals, eigvecs_b = linalg.eigh(S_b_white)
                eigvecs = whiten @ eigvecs_b

            # Floor eigenvalues for numerical stability
            eigvals = np.maximum(eigvals, self.config.min_eigenvalue)

            # Select top-K dimensions (descending order)
            K = min(self.config.latent_dim, D, num_classes - 1)
            idx = np.argsort(eigvals)[::-1][:K]

            # Whitening transform: D -> K
            # Includes both dimension reduction and variance normalization
            transform = eigvecs[:, idx]

            # Normalize columns
            norms = np.linalg.norm(transform, axis=0)
            norms = np.maximum(norms, 1e-10)
            transform = transform / norms

            # Step 6: Compute covariances in latent space
            # Project scatter matrices
            phi_b = transform.T @ S_b @ transform
            phi_w = transform.T @ S_w @ transform

            # Regularize in latent space
            phi_b = self._ensure_positive_definite(phi_b)
            phi_w = self._ensure_positive_definite(phi_w)

            # Step 7: Precompute scoring matrices
            phi_b_inv = linalg.inv(phi_b)
            phi_w_inv = linalg.inv(phi_w)

            # λ = (Φ_b^-1 + Φ_w^-1)^-1
            lambda_mat = linalg.inv(phi_b_inv + phi_w_inv)

            # γ = (Φ_b^-1 + 2*Φ_w^-1)^-1 (for same-identity pairs)
            gamma_mat = linalg.inv(phi_b_inv + 2 * phi_w_inv)

            # Log-determinants for normalization
            _, log_det_lambda = np.linalg.slogdet(lambda_mat)
            _, log_det_gamma = np.linalg.slogdet(gamma_mat)
            _, log_det_phi_b = np.linalg.slogdet(phi_b)

            # Step 8: Build identity models
            identity_embeddings = {}
            identity_centroids = {}
            identity_latent = {}

            for lbl in unique_labels:
                mask = [l == lbl for l in labels]
                class_embs = embeddings[mask]
                identity_embeddings[lbl] = [e for e in class_embs]
                identity_centroids[lbl] = class_embs.mean(axis=0)

                # Precompute latent representation
                centered = identity_centroids[lbl] - global_mean
                identity_latent[lbl] = centered @ transform

            # Create model
            self.model = PLDAModel(
                global_mean=global_mean,
                whiten_transform=transform,
                phi_b=phi_b,
                phi_w=phi_w,
                phi_b_inv=phi_b_inv,
                phi_w_inv=phi_w_inv,
                lambda_mat=lambda_mat,
                gamma_mat=gamma_mat,
                log_det_lambda=float(log_det_lambda),
                log_det_gamma=float(log_det_gamma),
                log_det_phi_b=float(log_det_phi_b),
                identity_embeddings=identity_embeddings,
                identity_centroids=identity_centroids,
                identity_latent=identity_latent
            )

            # Step 9: Calibrate LLR scores
            self._calibrate_from_training(embeddings, labels)

            self._trained = True
            logger.info(
                f"PLDA training complete: latent_dim={K}, "
                f"{len(identity_centroids)} identity models"
            )

            return True

        except Exception as e:
            logger.error(f"PLDA training failed: {e}", exc_info=True)
            return False

    def _ensure_positive_definite(self, mat: np.ndarray) -> np.ndarray:
        """Ensure matrix is positive definite by flooring eigenvalues."""
        eigvals, eigvecs = linalg.eigh(mat)
        eigvals = np.maximum(eigvals, self.config.min_eigenvalue)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def _calibrate_from_training(
        self,
        embeddings: np.ndarray,
        labels: List[str]
    ) -> None:
        """Calibrate LLR scores using training data.

        Computes mean/std of impostor scores so that:
        - Calibrated LLR ≈ 0 at impostor mean
        - Calibrated LLR in standard units
        """
        if self.model is None:
            return

        # Compute pairwise LLR scores for a subset
        unique_labels = list(set(labels))
        same_scores = []
        diff_scores = []

        # Sample pairs to avoid O(N^2) computation
        max_pairs = 500
        n_same = 0
        n_diff = 0

        label_to_embs = {}
        for emb, lbl in zip(embeddings, labels):
            if lbl not in label_to_embs:
                label_to_embs[lbl] = []
            label_to_embs[lbl].append(emb)

        # Same-identity pairs
        for lbl, embs in label_to_embs.items():
            if len(embs) < 2:
                continue
            for i in range(min(len(embs) - 1, max_pairs // len(unique_labels))):
                for j in range(i + 1, min(len(embs), i + 3)):
                    llr = self._compute_llr_pair(embs[i], embs[j])
                    if np.isfinite(llr):
                        same_scores.append(llr)
                        n_same += 1
                        if n_same >= max_pairs:
                            break
                if n_same >= max_pairs:
                    break
            if n_same >= max_pairs:
                break

        # Different-identity pairs
        labels_list = list(label_to_embs.keys())
        for i, lbl1 in enumerate(labels_list):
            for lbl2 in labels_list[i+1:]:
                embs1 = label_to_embs[lbl1]
                embs2 = label_to_embs[lbl2]
                for e1 in embs1[:3]:
                    for e2 in embs2[:3]:
                        llr = self._compute_llr_pair(e1, e2)
                        if np.isfinite(llr):
                            diff_scores.append(llr)
                            n_diff += 1
                            if n_diff >= max_pairs:
                                break
                    if n_diff >= max_pairs:
                        break
                if n_diff >= max_pairs:
                    break
            if n_diff >= max_pairs:
                break

        if diff_scores:
            self.model.llr_mean = float(np.mean(diff_scores))
            all_scores = same_scores + diff_scores
            self.model.llr_std = float(np.std(all_scores)) if all_scores else 1.0
            self.model.llr_std = max(self.model.llr_std, 0.1)  # Prevent zero std

            logger.debug(
                f"LLR calibration: same_mean={np.mean(same_scores):.2f}, "
                f"diff_mean={self.model.llr_mean:.2f}, std={self.model.llr_std:.2f}"
            )

    def _project_to_latent(self, embedding: np.ndarray) -> np.ndarray:
        """Project embedding to PLDA latent space.

        Args:
            embedding: Raw embedding [D]

        Returns:
            Latent representation [K]
        """
        centered = embedding - self.model.global_mean
        return centered @ self.model.whiten_transform

    def _compute_llr_pair(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray
    ) -> float:
        """Compute log-likelihood ratio for a pair of embeddings.

        LLR = log P(emb1, emb2 | same identity) - log P(emb1, emb2 | different)

        This is the core PLDA scoring formula from Prince & Elder.

        Args:
            emb1: First embedding [D]
            emb2: Second embedding [D]

        Returns:
            Log-likelihood ratio (higher = more likely same person)
        """
        if self.model is None:
            return 0.0

        # Project to latent space
        x1 = self._project_to_latent(emb1)
        x2 = self._project_to_latent(emb2)

        m = self.model

        # Same-identity hypothesis H_s:
        # P(x1, x2 | H_s) involves integrating over shared identity
        # = ∫ P(x1 | h) P(x2 | h) P(h) dh
        #
        # Different-identity hypothesis H_d:
        # P(x1, x2 | H_d) = P(x1) P(x2)
        #
        # After Gaussian integration, LLR has closed form:
        # LLR = 0.5 * (x1 + x2)^T γ (x1 + x2)
        #     - 0.5 * x1^T λ x1
        #     - 0.5 * x2^T λ x2
        #     + 0.5 * (log|γ| - log|λ|)

        sum_vec = x1 + x2

        # Quadratic terms
        term_same = sum_vec @ m.gamma_mat @ sum_vec
        term_x1 = x1 @ m.lambda_mat @ x1
        term_x2 = x2 @ m.lambda_mat @ x2

        # LLR computation
        llr = 0.5 * term_same - 0.5 * term_x1 - 0.5 * term_x2

        # Add log-determinant normalization
        llr += 0.5 * (m.log_det_gamma - m.log_det_lambda)

        return float(llr)

    def score(
        self,
        probe: np.ndarray,
        target_person_id: str
    ) -> float:
        """Compute PLDA LLR score for probe vs enrolled identity.

        Uses the centroid of enrolled embeddings for the target identity.
        For low-dimensional PLDA (few identities), augments with cosine similarity.

        Args:
            probe: Probe embedding [D]
            target_person_id: Target identity to compare against

        Returns:
            Log-likelihood ratio score (higher = more likely same person)
        """
        if not self._trained or self.model is None:
            logger.warning("PLDA not trained, returning -inf")
            return float('-inf')

        if target_person_id not in self.model.identity_centroids:
            logger.debug(f"Unknown identity: {target_person_id}")
            return float('-inf')

        try:
            # Basic embedding validation
            probe_norm = np.linalg.norm(probe)
            if probe_norm < 0.1 or probe_norm > 10.0:
                logger.debug(f"Probe embedding has unusual norm: {probe_norm:.3f}")
                return float('-inf')

            # Get enrolled centroid
            target = self.model.identity_centroids[target_person_id]

            # Check if PLDA has enough dimensions for reliable scoring
            latent_dim = self.model.whiten_transform.shape[1]

            if latent_dim <= 2:
                # Low-dimensional case: PLDA has limited discriminative power
                # Use cosine similarity as primary signal, augmented with PLDA
                target_norm = np.linalg.norm(target)
                if target_norm > 1e-6 and probe_norm > 1e-6:
                    cosine_sim = np.dot(probe, target) / (probe_norm * target_norm)
                else:
                    cosine_sim = 0.0

                # Convert cosine similarity to LLR-like score
                # cosine=1.0 -> high positive LLR, cosine=0.5 -> ~0, cosine<0.5 -> negative
                # Using logit transform: LLR = log(p/(1-p)) where p = (cosine+1)/2
                p = np.clip((cosine_sim + 1) / 2, 0.01, 0.99)
                cosine_llr = np.log(p / (1 - p))

                # Scale by enrollment count
                n_enroll = len(self.model.identity_embeddings.get(target_person_id, []))
                reliability = min(n_enroll, 10) / 10.0
                cosine_llr *= (0.6 + 0.4 * reliability)

                return float(cosine_llr)

            # Normal case: Use PLDA LLR
            raw_llr = self._compute_llr_pair(probe, target)

            # Apply calibration only if we have meaningful separation
            if self.model.llr_std > 0.5:
                calibrated_llr = (raw_llr - self.model.llr_mean) / self.model.llr_std
            else:
                calibrated_llr = raw_llr - self.model.llr_mean

            # Scale by enrollment count
            n_enroll = len(self.model.identity_embeddings.get(target_person_id, []))
            reliability = min(n_enroll, 10) / 10.0
            calibrated_llr *= (0.6 + 0.4 * reliability)

            return float(calibrated_llr)

        except Exception as e:
            logger.error(f"PLDA scoring failed: {e}")
            return float('-inf')

    def score_against_all(
        self,
        probe: np.ndarray,
        min_llr: float = float('-inf')
    ) -> Tuple[Optional[str], float, float, Dict[str, float]]:
        """Score probe against all enrolled identities.

        Args:
            probe: Probe embedding [D]
            min_llr: Minimum LLR threshold (skip identities below this)

        Returns:
            Tuple of (best_id, best_llr, margin, all_scores)
        """
        if not self._trained or self.model is None:
            return None, float('-inf'), 0.0, {}

        all_scores = {}
        for person_id in self.model.identity_centroids:
            llr = self.score(probe, person_id)
            if llr > min_llr:
                all_scores[person_id] = llr

        if not all_scores:
            return None, float('-inf'), 0.0, {}

        # Sort by score
        sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

        best_id, best_llr = sorted_scores[0]
        second_llr = sorted_scores[1][1] if len(sorted_scores) > 1 else float('-inf')
        margin = best_llr - second_llr

        return best_id, best_llr, margin, all_scores

    def score_batch(
        self,
        probe: np.ndarray,
        candidate_ids: List[str]
    ) -> Dict[str, float]:
        """Score probe against multiple candidate identities.

        Args:
            probe: Probe embedding [D]
            candidate_ids: List of candidate identity IDs

        Returns:
            Dict mapping person_id to LLR score
        """
        return {pid: self.score(probe, pid) for pid in candidate_ids}

    def score_with_uncertainty(
        self,
        probe: np.ndarray,
        target_person_id: str
    ) -> Tuple[float, float]:
        """Score probe against enrolled identity with uncertainty estimate.

        Computes LLR against each enrolled embedding individually,
        then returns mean and variance to quantify uncertainty.

        Useful when enrollment embeddings are dispersed.

        Args:
            probe: Probe embedding [D]
            target_person_id: Target identity

        Returns:
            (mean_llr, uncertainty)
        """
        if not self._trained or self.model is None:
            return float('-inf'), float('inf')

        enrollments = self.model.identity_embeddings.get(target_person_id, [])
        if not enrollments:
            return float('-inf'), float('inf')

        # Check if using cosine fallback
        latent_dim = self.model.whiten_transform.shape[1]
        use_cosine = latent_dim <= 2

        probe_norm = np.linalg.norm(probe)
        if probe_norm < 1e-6:
            return float('-inf'), float('inf')

        # Score against each enrolled embedding
        llrs = []
        for enrolled in enrollments:
            if use_cosine:
                # Use cosine similarity based scoring
                enrolled_norm = np.linalg.norm(enrolled)
                if enrolled_norm > 1e-6:
                    cosine_sim = np.dot(probe, enrolled) / (probe_norm * enrolled_norm)
                    p = np.clip((cosine_sim + 1) / 2, 0.01, 0.99)
                    llr = np.log(p / (1 - p))
                    llrs.append(llr)
            else:
                # Use PLDA LLR
                raw_llr = self._compute_llr_pair(probe, enrolled)
                if np.isfinite(raw_llr):
                    llrs.append(raw_llr)

        if not llrs:
            return float('-inf'), float('inf')

        llrs = np.array(llrs)

        # Apply calibration for PLDA (not needed for cosine which is already calibrated)
        if not use_cosine:
            if self.model.llr_std > 0.5:
                llrs = (llrs - self.model.llr_mean) / self.model.llr_std
            else:
                llrs = llrs - self.model.llr_mean

        mean_llr = float(np.mean(llrs))
        uncertainty = float(np.std(llrs)) if len(llrs) > 1 else 0.0

        return mean_llr, uncertainty

    def get_decision(
        self,
        probe: np.ndarray,
        candidate_ids: List[str],
        llr_threshold: float = 0.0,
        margin_threshold: float = 0.5
    ) -> Tuple[Optional[str], float, float]:
        """Get PLDA-based identity decision with thresholds.

        Args:
            probe: Probe embedding
            candidate_ids: Candidate identities from FAISS Stage 1
            llr_threshold: Minimum LLR for acceptance
            margin_threshold: Minimum LLR margin between top two

        Returns:
            (best_person_id, best_llr, margin)
            Returns (None, llr, margin) if thresholds not met
        """
        if not candidate_ids:
            return None, float('-inf'), 0.0

        scores = self.score_batch(probe, candidate_ids)

        if not scores:
            return None, float('-inf'), 0.0

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        best_id, best_llr = sorted_scores[0]
        second_llr = sorted_scores[1][1] if len(sorted_scores) > 1 else float('-inf')
        margin = best_llr - second_llr

        # Apply thresholds
        if best_llr < llr_threshold:
            return None, best_llr, margin

        if margin < margin_threshold and len(sorted_scores) > 1:
            # Margin too small, not confident
            return None, best_llr, margin

        return best_id, best_llr, margin

    def llr_to_probability(self, llr: float) -> float:
        """Convert calibrated LLR to posterior probability.

        Assuming equal priors P(same) = P(diff) = 0.5:
            P(same | score) = sigmoid(LLR) = 1 / (1 + exp(-LLR))

        Args:
            llr: Calibrated LLR score

        Returns:
            Probability in [0, 1]
        """
        # Clip to prevent overflow
        llr_clipped = np.clip(llr, -20, 20)
        prob = 1.0 / (1.0 + np.exp(-llr_clipped))
        return float(prob)

    def calibrate_score(
        self,
        llr: float,
        target_range: Tuple[float, float] = (0.0, 1.0)
    ) -> float:
        """Calibrate LLR to target score range.

        Uses sigmoid calibration for smooth mapping.

        Args:
            llr: Raw or calibrated LLR score
            target_range: Target output range (min, max)

        Returns:
            Calibrated score in target range
        """
        # Sigmoid with moderate slope
        k = 1.0
        clipped_llr = np.clip(-k * llr, -500, 500)
        sigmoid = 1.0 / (1.0 + np.exp(clipped_llr))

        # Map to target range
        min_out, max_out = target_range
        calibrated = min_out + (max_out - min_out) * sigmoid

        # Cap at 0.98 to avoid overconfidence
        return float(min(calibrated, 0.98))

    def compute_unknown_score(
        self,
        probe: np.ndarray,
        unknown_threshold: float = 0.0
    ) -> float:
        """Compute probability that probe is an unknown person.

        If even the best LLR is below threshold, the person is likely unknown.

        Args:
            probe: Probe embedding [D]
            unknown_threshold: LLR threshold below which person is "unknown"

        Returns:
            Probability of unknown in [0, 1]
        """
        best_id, best_llr, margin, _ = self.score_against_all(probe)

        if best_id is None:
            return 1.0

        # P(unknown) based on best LLR relative to threshold
        # If best_llr >> threshold: low unknown probability
        # If best_llr << threshold: high unknown probability
        p_same = self.llr_to_probability(best_llr - unknown_threshold)
        return 1.0 - p_same

    def save(self, path: str) -> bool:
        """Save PLDA model to file.

        Args:
            path: Output file path (.npz format)

        Returns:
            True if successful
        """
        if not self._trained or self.model is None:
            logger.error("Cannot save untrained PLDA model")
            return False

        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            # Save numpy arrays
            np.savez(
                path,
                global_mean=self.model.global_mean,
                whiten_transform=self.model.whiten_transform,
                phi_b=self.model.phi_b,
                phi_w=self.model.phi_w,
                phi_b_inv=self.model.phi_b_inv,
                phi_w_inv=self.model.phi_w_inv,
                lambda_mat=self.model.lambda_mat,
                gamma_mat=self.model.gamma_mat,
                log_det_lambda=self.model.log_det_lambda,
                log_det_gamma=self.model.log_det_gamma,
                log_det_phi_b=self.model.log_det_phi_b,
                llr_mean=self.model.llr_mean,
                llr_std=self.model.llr_std,
                # Config
                embedding_dim=self.config.embedding_dim,
                latent_dim=self.config.latent_dim,
                between_class_reg=self.config.between_class_reg,
                within_class_reg=self.config.within_class_reg
            )

            # Save identity models as JSON
            models_path = Path(path).with_suffix('.models.json')
            models_data = {
                person_id: {
                    'centroid': self.model.identity_centroids[person_id].tolist(),
                    'embeddings': [e.tolist() for e in embs],
                    'count': len(embs)
                }
                for person_id, embs in self.model.identity_embeddings.items()
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

            # Update config
            self.config.embedding_dim = int(data['embedding_dim'])

            # Handle both old and new format
            if 'latent_dim' in data:
                self.config.latent_dim = int(data['latent_dim'])
            elif 'plda_dim' in data:
                self.config.latent_dim = int(data['plda_dim'])

            if 'between_class_reg' in data:
                self.config.between_class_reg = float(data['between_class_reg'])
            if 'within_class_reg' in data:
                self.config.within_class_reg = float(data['within_class_reg'])

            # Check if this is the new format or old format
            if 'whiten_transform' in data:
                # New format
                self.model = PLDAModel(
                    global_mean=data['global_mean'],
                    whiten_transform=data['whiten_transform'],
                    phi_b=data['phi_b'],
                    phi_w=data['phi_w'],
                    phi_b_inv=data['phi_b_inv'],
                    phi_w_inv=data['phi_w_inv'],
                    lambda_mat=data['lambda_mat'],
                    gamma_mat=data['gamma_mat'],
                    log_det_lambda=float(data['log_det_lambda']),
                    log_det_gamma=float(data['log_det_gamma']),
                    log_det_phi_b=float(data['log_det_phi_b']),
                    llr_mean=float(data['llr_mean']),
                    llr_std=float(data['llr_std'])
                )
            else:
                # Old format - need to recompute matrices
                logger.info("Loading old PLDA format, recomputing matrices...")
                global_mean = data['global_mean']
                between_cov = data['between_cov']
                within_cov = data['within_cov']

                # Compute transform using old covariances
                D = len(global_mean)
                reg = float(data.get('regularization', 1e-5))

                # Simplified whitening
                eigvals, eigvecs = linalg.eigh(within_cov + reg * np.eye(D))
                eigvals = np.maximum(eigvals, 1e-6)
                K = min(self.config.latent_dim, D)
                idx = np.argsort(eigvals)[::-1][:K]
                transform = eigvecs[:, idx] @ np.diag(1.0 / np.sqrt(eigvals[idx]))

                phi_b = transform.T @ between_cov @ transform
                phi_w = np.eye(K)

                phi_b_inv = linalg.inv(phi_b + 1e-5 * np.eye(K))
                phi_w_inv = np.eye(K)
                lambda_mat = linalg.inv(phi_b_inv + phi_w_inv)
                gamma_mat = linalg.inv(phi_b_inv + 2 * phi_w_inv)

                _, log_det_lambda = np.linalg.slogdet(lambda_mat)
                _, log_det_gamma = np.linalg.slogdet(gamma_mat)
                _, log_det_phi_b = np.linalg.slogdet(phi_b)

                self.model = PLDAModel(
                    global_mean=global_mean,
                    whiten_transform=transform,
                    phi_b=phi_b,
                    phi_w=phi_w,
                    phi_b_inv=phi_b_inv,
                    phi_w_inv=phi_w_inv,
                    lambda_mat=lambda_mat,
                    gamma_mat=gamma_mat,
                    log_det_lambda=float(log_det_lambda),
                    log_det_gamma=float(log_det_gamma),
                    log_det_phi_b=float(log_det_phi_b),
                    llr_mean=0.0,
                    llr_std=1.0
                )

            # Load identity models
            models_path = Path(path).with_suffix('.models.json')
            if models_path.exists():
                with open(models_path, 'r') as f:
                    models_data = json.load(f)

                self.model.identity_embeddings = {}
                self.model.identity_centroids = {}
                self.model.identity_latent = {}

                for person_id, info in models_data.items():
                    # Handle both old format (embedding) and new format (centroid)
                    if 'centroid' in info:
                        self.model.identity_centroids[person_id] = np.array(info['centroid'])
                    elif 'embedding' in info:
                        self.model.identity_centroids[person_id] = np.array(info['embedding'])

                    if 'embeddings' in info:
                        self.model.identity_embeddings[person_id] = [
                            np.array(e) for e in info['embeddings']
                        ]
                    else:
                        # Old format - only has single embedding
                        self.model.identity_embeddings[person_id] = [
                            self.model.identity_centroids[person_id]
                        ]

                    # Recompute latent representation
                    centered = self.model.identity_centroids[person_id] - self.model.global_mean
                    self.model.identity_latent[person_id] = centered @ self.model.whiten_transform
            else:
                logger.warning(f"Identity models file not found: {models_path}")
                self.model.identity_embeddings = {}
                self.model.identity_centroids = {}
                self.model.identity_latent = {}

            self._trained = True
            logger.info(
                f"Loaded PLDA model from {path}: "
                f"{len(self.model.identity_centroids)} identities, "
                f"latent_dim={self.model.whiten_transform.shape[1]}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load PLDA model: {e}", exc_info=True)
            return False

    def __repr__(self) -> str:
        if self.model is None:
            return f"PLDAScorer(trained=False)"
        return (
            f"PLDAScorer("
            f"embedding_dim={self.config.embedding_dim}, "
            f"latent_dim={self.model.whiten_transform.shape[1] if self.model.whiten_transform is not None else 0}, "
            f"identities={len(self.model.identity_centroids)}, "
            f"trained={self._trained})"
        )
