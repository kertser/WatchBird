"""Person classification using CLIP for zero-shot recognition.

Classifies detected persons as:
- Unarmed civilian
- Armed civilian
- IDF soldier (OD green uniform)
"""

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Suppress noisy warnings from transformers/huggingface BEFORE importing
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["ACCELERATE_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore", message=".*CLIPImageProcessor.*")
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*not sharded.*")
warnings.filterwarnings("ignore", message=".*layers were not.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=UserWarning, module="accelerate")

# Try to import CLIP dependencies
try:
    # Redirect stdout temporarily to suppress model loading report
    import torch
    from PIL import Image

    # Suppress transformers and huggingface_hub logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True

    # Check for DirectML support (AMD/Intel GPU on Windows)
    try:
        import torch_directml
        DIRECTML_AVAILABLE = torch_directml.is_available()
    except ImportError:
        DIRECTML_AVAILABLE = False
except ImportError:
    CLIP_AVAILABLE = False
    DIRECTML_AVAILABLE = False
    logger.warning("CLIP dependencies not available. Install with: uv pip install transformers torch pillow")


class PersonClassifier:
    """Classify persons using CLIP zero-shot classification.

    Distinguishes between:
    - unarmed_civilian: Person without weapons, casual clothing
    - armed_civilian: Person with visible weapon, non-military clothing
    - soldier: Person in military uniform (IDF OD green)
    """

    # Classification categories with descriptive prompts
    CATEGORIES = {
        "soldier": [
            "a soldier wearing olive drab military uniform",
            "a person in IDF military combat uniform",
            "a soldier wearing green military fatigues",
            "military personnel in olive green uniform with gear",
            "an armed soldier in combat uniform",
        ],
        "armed_civilian": [
            "a civilian person holding a weapon",
            "a person in casual clothes carrying a gun",
            "an armed person not in military uniform",
            "a civilian with a rifle or pistol",
            "a person in street clothes holding a firearm",
        ],
        "unarmed_civilian": [
            "an unarmed civilian person",
            "a person in casual civilian clothing",
            "a regular person without weapons",
            "a civilian in normal street clothes",
            "an ordinary person walking",
        ],
    }

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize CLIP classifier.

        Args:
            model_name: HuggingFace CLIP model name
            device: Device to use (cuda, cpu, or auto-detect)
            cache_dir: Directory to cache model weights
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or "models/clip_cache"

        # Auto-detect device
        if device is None:
            if CLIP_AVAILABLE:
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif DIRECTML_AVAILABLE:
                    self.device = torch_directml.device()
                else:
                    self.device = "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self._loaded = False

        # Pre-computed text embeddings for each category
        self._text_embeddings: Optional[torch.Tensor] = None
        self._category_names: List[str] = list(self.CATEGORIES.keys())

    def load(self) -> bool:
        """Load CLIP model and pre-compute text embeddings.

        Returns:
            True if successful, False otherwise
        """
        if not CLIP_AVAILABLE:
            logger.error("CLIP dependencies not installed")
            return False

        try:
            logger.info(f"Loading CLIP model: {self.model_name}")

            # Create cache directory
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

            # Suppress stdout/stderr during model loading (LOAD REPORT, sharding warnings)
            import io
            import sys
            import builtins

            # Create a custom context manager that fully suppresses output
            class SuppressOutput:
                def __enter__(self):
                    self._stdout = sys.stdout
                    self._stderr = sys.stderr
                    self._print = builtins.print
                    sys.stdout = io.StringIO()
                    sys.stderr = io.StringIO()
                    builtins.print = lambda *args, **kwargs: None
                    return self

                def __exit__(self, *args):
                    sys.stdout = self._stdout
                    sys.stderr = self._stderr
                    builtins.print = self._print

            # Check if model is already cached locally
            model_cache_path = Path(self.cache_dir) / f"models--{self.model_name.replace('/', '--')}"
            local_files_only = model_cache_path.exists()

            if local_files_only:
                logger.info(f"Loading CLIP model from local cache: {self.cache_dir}")
            else:
                logger.info(f"Downloading CLIP model to: {self.cache_dir}")

            # Try to load with safetensors format (required for newer transformers with torch < 2.6)
            # The PR revision has safetensors available
            try:
                with SuppressOutput():
                    model = CLIPModel.from_pretrained(
                        self.model_name,
                        cache_dir=self.cache_dir,
                        revision="refs/pr/66",  # This revision has safetensors
                        use_safetensors=True,
                        local_files_only=local_files_only
                    )
                with SuppressOutput():
                    self.model = model.to(self.device)
            except Exception as e:
                if local_files_only:
                    # Try without local_files_only restriction
                    logger.info("Local cache outdated, downloading update...")
                    with SuppressOutput():
                        model = CLIPModel.from_pretrained(
                            self.model_name,
                            cache_dir=self.cache_dir,
                            revision="refs/pr/66",
                            use_safetensors=True,
                            local_files_only=False
                        )
                    with SuppressOutput():
                        self.model = model.to(self.device)
                else:
                    logger.warning(f"Safetensors load failed, trying default: {e}")
                    # Fallback: try without safetensors
                    with SuppressOutput():
                        model = CLIPModel.from_pretrained(
                            self.model_name,
                            cache_dir=self.cache_dir,
                        )
                    with SuppressOutput():
                        self.model = model.to(self.device)

            with SuppressOutput():
                self.processor = CLIPProcessor.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    use_fast=False,  # Suppress fast processor warning
                    local_files_only=local_files_only
                )

            # Pre-compute text embeddings for all prompts
            self._precompute_text_embeddings()

            self._loaded = True
            logger.info(f"CLIP classifier loaded - device: {self.device}")
            logger.info(f"Categories: {', '.join(self._category_names)}")
            return True

        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            return False

    def _precompute_text_embeddings(self) -> None:
        """Pre-compute and cache text embeddings for all category prompts."""
        all_prompts = []
        prompt_to_category = []

        for category, prompts in self.CATEGORIES.items():
            for prompt in prompts:
                all_prompts.append(prompt)
                prompt_to_category.append(category)

        # Encode all prompts
        inputs = self.processor(
            text=all_prompts,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            text_output = self.model.get_text_features(**inputs)
            # Handle both old API (tensor) and new API (BaseModelOutputWithPooling)
            if hasattr(text_output, 'pooler_output'):
                text_features = text_output.pooler_output
            else:
                text_features = text_output
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Average embeddings per category
        category_embeddings = []
        for category in self._category_names:
            indices = [i for i, c in enumerate(prompt_to_category) if c == category]
            cat_embedding = text_features[indices].mean(dim=0)
            cat_embedding = cat_embedding / cat_embedding.norm()
            category_embeddings.append(cat_embedding)

        self._text_embeddings = torch.stack(category_embeddings)
        self._prompt_to_category = prompt_to_category

    @property
    def loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def classify(
        self,
        person_crop: np.ndarray,
        return_all_scores: bool = False
    ) -> Tuple[str, float, Optional[Dict[str, float]]]:
        """Classify a cropped person image.

        Args:
            person_crop: BGR image of cropped person
            return_all_scores: Whether to return scores for all categories

        Returns:
            Tuple of (category, confidence, optional_all_scores)
            category: "soldier", "armed_civilian", or "unarmed_civilian"
            confidence: Score for the predicted category (0-1)
            all_scores: Dict of all category scores if return_all_scores=True
        """
        if not self._loaded:
            return "unarmed_civilian", 0.0, None

        # Validate input
        if person_crop is None or person_crop.size == 0:
            return "unarmed_civilian", 0.0, None

        # Minimum size check
        if person_crop.shape[0] < 32 or person_crop.shape[1] < 32:
            return "unarmed_civilian", 0.0, None

        try:
            # Convert BGR to RGB PIL Image
            rgb_image = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Process image
            inputs = self.processor(
                images=pil_image,
                return_tensors="pt"
            ).to(self.device)

            # Get image features
            with torch.no_grad():
                image_output = self.model.get_image_features(**inputs)
                # Handle both old API (tensor) and new API (BaseModelOutputWithPooling)
                if hasattr(image_output, 'pooler_output'):
                    image_features = image_output.pooler_output
                else:
                    image_features = image_output
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Compute similarity with pre-computed text embeddings
                similarities = (image_features @ self._text_embeddings.T).squeeze(0)

                # Apply softmax to get probabilities
                probs = torch.softmax(similarities * 100, dim=0)  # Scale for sharper distribution

            # Get results
            probs_np = probs.cpu().numpy()
            best_idx = int(probs_np.argmax())
            best_category = self._category_names[best_idx]
            best_score = float(probs_np[best_idx])

            if return_all_scores:
                all_scores = {
                    cat: float(probs_np[i])
                    for i, cat in enumerate(self._category_names)
                }
                return best_category, best_score, all_scores

            return best_category, best_score, None

        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return "unarmed_civilian", 0.0, None

    def classify_batch(
        self,
        person_crops: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        """Classify multiple person crops in a batch.

        Args:
            person_crops: List of BGR images of cropped persons

        Returns:
            List of (category, confidence) tuples
        """
        if not self._loaded or not person_crops:
            return [("unarmed_civilian", 0.0)] * len(person_crops)

        try:
            # Convert all valid crops to PIL images
            pil_images = []
            valid_indices = []

            for i, crop in enumerate(person_crops):
                if crop is not None and crop.size > 0:
                    if crop.shape[0] >= 32 and crop.shape[1] >= 32:
                        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        pil_images.append(Image.fromarray(rgb))
                        valid_indices.append(i)

            if not pil_images:
                return [("unarmed_civilian", 0.0)] * len(person_crops)

            # Process batch
            inputs = self.processor(
                images=pil_images,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            with torch.no_grad():
                image_output = self.model.get_image_features(**inputs)
                # Handle both old API (tensor) and new API (BaseModelOutputWithPooling)
                if hasattr(image_output, 'pooler_output'):
                    image_features = image_output.pooler_output
                else:
                    image_features = image_output
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                similarities = image_features @ self._text_embeddings.T
                probs = torch.softmax(similarities * 100, dim=1)

            probs_np = probs.cpu().numpy()

            # Build results
            results = [("unarmed_civilian", 0.0)] * len(person_crops)
            for batch_idx, orig_idx in enumerate(valid_indices):
                best_idx = int(probs_np[batch_idx].argmax())
                results[orig_idx] = (
                    self._category_names[best_idx],
                    float(probs_np[batch_idx, best_idx])
                )

            return results

        except Exception as e:
            logger.warning(f"Batch classification failed: {e}")
            return [("unarmed_civilian", 0.0)] * len(person_crops)


def get_person_type_color(person_type: str) -> Tuple[int, int, int]:
    """Get visualization color for person type.

    Args:
        person_type: Classification result

    Returns:
        BGR color tuple
    """
    colors = {
        "soldier": (0, 180, 0),         # Green (friendly military)
        "armed_civilian": (0, 0, 255),   # Red (threat)
        "unarmed_civilian": (255, 180, 0),  # Cyan/light blue (neutral)
    }
    return colors.get(person_type, (128, 128, 128))


def get_person_type_label(person_type: str, confidence: float) -> str:
    """Get display label for person type.

    Args:
        person_type: Classification result
        confidence: Confidence score

    Returns:
        Display label string
    """
    # Use elegant symbols and short labels
    labels = {
        "soldier": "IDF",           # Israeli Defense Forces
        "armed_civilian": "ARMED",   # Armed threat
        "unarmed_civilian": "CIV",   # Civilian
    }
    label = labels.get(person_type, "?")
    return f"{label} {confidence:.0%}"


def get_person_type_icon(person_type: str) -> str:
    """Get icon/symbol for person type.

    Args:
        person_type: Classification result

    Returns:
        Unicode symbol
    """
    icons = {
        "soldier": "[S]",      # Shield/soldier
        "armed_civilian": "[!]",  # Warning
        "unarmed_civilian": "[C]",   # Person
    }
    return icons.get(person_type, "[?]")
