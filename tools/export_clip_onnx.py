"""Export and quantize CLIP model to ONNX format.

This creates a lightweight ONNX version of CLIP for person classification:
- clip_vision.onnx: Full precision vision encoder (~350MB)
- clip_vision_int8.onnx: INT8 quantized version (~150MB)
- clip_text_embeddings.npy: Pre-computed text embeddings for classes

Usage:
    python tools/export_clip_onnx.py
    python tools/export_clip_onnx.py --no-quantize  # Skip INT8 quantization
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

# Suppress warnings during export
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

import numpy as np


def export_clip_to_onnx(
    output_dir: str = "models",
    quantize: bool = True,
    opset_version: int = 14
) -> str:
    """Export CLIP ViT-B/32 to ONNX with optional INT8 quantization.

    Args:
        output_dir: Directory to save models
        quantize: Whether to create INT8 quantized version
        opset_version: ONNX opset version

    Returns:
        Path to the final model (quantized if enabled)
    """
    import torch
    from transformers import CLIPModel, CLIPProcessor

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CLIP to ONNX Export Tool")
    print("=" * 60)

    print("\n[1/4] Loading CLIP model from HuggingFace...")

    # Use safetensors format to avoid torch.load security issue
    try:
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            revision="refs/pr/66",  # This revision has safetensors
            use_safetensors=True,
            attn_implementation="eager"  # Use eager attention for ONNX compatibility
        )
    except Exception as e:
        print(f"    Safetensors failed, trying default: {e}")
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            attn_implementation="eager"
        )

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    # Create wrapper for vision encoder + projection
    class CLIPVisionWrapper(torch.nn.Module):
        """Wrapper that combines vision encoder and projection layer."""

        def __init__(self, vision_model, projection):
            super().__init__()
            self.vision_model = vision_model
            self.projection = projection

        def forward(self, pixel_values):
            # Get vision model output
            outputs = self.vision_model(pixel_values=pixel_values)
            pooled = outputs.pooler_output

            # Project to shared embedding space
            projected = self.projection(pooled)

            # L2 normalize
            projected = projected / projected.norm(dim=-1, keepdim=True)
            return projected

    wrapper = CLIPVisionWrapper(model.vision_model, model.visual_projection)
    wrapper.eval()

    # Create dummy input (batch of 224x224 RGB images)
    dummy_input = torch.randn(1, 3, 224, 224)

    onnx_path = output_dir / "clip_vision.onnx"
    print(f"\n[2/4] Exporting to ONNX: {onnx_path}")

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["image_features"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "image_features": {0: "batch_size"}
            },
            opset_version=opset_version,
            do_constant_folding=True
        )

    onnx_size = onnx_path.stat().st_size / 1024 / 1024
    print(f"    Saved: {onnx_path} ({onnx_size:.1f} MB)")

    # Export text embeddings for the classification categories
    print(f"\n[3/4] Computing text embeddings for classification...")
    export_text_embeddings(model, processor, output_dir)

    final_model = str(onnx_path)

    if quantize:
        print(f"\n[4/4] Quantizing to INT8...")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantized_path = output_dir / "clip_vision_int8.onnx"

            quantize_dynamic(
                str(onnx_path),
                str(quantized_path),
                weight_type=QuantType.QInt8
            )

            quant_size = quantized_path.stat().st_size / 1024 / 1024
            print(f"    Saved: {quantized_path} ({quant_size:.1f} MB)")
            print(f"    Size reduction: {onnx_size:.1f} MB → {quant_size:.1f} MB ({(1 - quant_size/onnx_size)*100:.0f}% smaller)")

            final_model = str(quantized_path)

        except ImportError:
            print("    WARNING: onnxruntime.quantization not available")
            print("    Install with: pip install onnxruntime")
            print("    Skipping quantization...")
    else:
        print(f"\n[4/4] Skipping quantization (--no-quantize)")

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"\nFiles created in {output_dir}/:")
    print(f"  - clip_vision.onnx ({onnx_size:.1f} MB)")
    if quantize and Path(output_dir / "clip_vision_int8.onnx").exists():
        print(f"  - clip_vision_int8.onnx ({quant_size:.1f} MB) ← Use this one")
    print(f"  - clip_text_embeddings.npy")

    return final_model


def export_text_embeddings(model, processor, output_dir: Path):
    """Pre-compute and save text embeddings for classification categories.

    Uses multiple prompts per category for more robust classification.
    """
    import torch

    # Classification prompts - same as PersonClassifier
    categories = {
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

    category_names = list(categories.keys())
    all_prompts = []
    prompt_to_category = []

    for category, prompts in categories.items():
        for prompt in prompts:
            all_prompts.append(prompt)
            prompt_to_category.append(category)

    # Compute text embeddings
    inputs = processor(text=all_prompts, return_tensors="pt", padding=True)

    with torch.no_grad():
        text_output = model.get_text_features(**inputs)
        if hasattr(text_output, 'pooler_output'):
            text_features = text_output.pooler_output
        else:
            text_features = text_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Average embeddings per category
    category_embeddings = []
    for category in category_names:
        indices = [i for i, c in enumerate(prompt_to_category) if c == category]
        cat_embedding = text_features[indices].mean(dim=0)
        cat_embedding = cat_embedding / cat_embedding.norm()
        category_embeddings.append(cat_embedding.numpy())

    embeddings_array = np.stack(category_embeddings)

    # Save embeddings
    embeddings_path = output_dir / "clip_text_embeddings.npy"
    np.save(str(embeddings_path), embeddings_array)
    print(f"    Saved: {embeddings_path} (shape: {embeddings_array.shape})")

    # Also save category names for reference
    labels_path = output_dir / "clip_labels.txt"
    with open(labels_path, "w") as f:
        for name in category_names:
            f.write(f"{name}\n")
    print(f"    Saved: {labels_path}")


def verify_onnx_model(model_path: str, text_embeddings_path: str):
    """Verify the exported ONNX model works correctly."""
    import onnxruntime as ort

    print(f"\nVerifying ONNX model: {model_path}")

    # Load model
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    # Create dummy input
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Run inference
    outputs = session.run(["image_features"], {"pixel_values": dummy_input})
    image_features = outputs[0]

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {image_features.shape}")
    print(f"  Output norm: {np.linalg.norm(image_features):.4f} (should be ~1.0)")

    # Load text embeddings
    text_embeddings = np.load(text_embeddings_path)
    print(f"  Text embeddings shape: {text_embeddings.shape}")

    # Compute similarities
    similarities = image_features @ text_embeddings.T
    print(f"  Similarities: {similarities[0]}")

    print("  ✓ ONNX model verified successfully!")


def main():
    parser = argparse.ArgumentParser(description="Export CLIP to ONNX")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--no-quantize", action="store_true", help="Skip INT8 quantization")
    parser.add_argument("--verify", action="store_true", help="Verify exported model")
    args = parser.parse_args()

    # Export
    final_model = export_clip_to_onnx(
        output_dir=args.output_dir,
        quantize=not args.no_quantize
    )

    # Verify
    if args.verify:
        text_embeddings_path = Path(args.output_dir) / "clip_text_embeddings.npy"
        verify_onnx_model(final_model, str(text_embeddings_path))


if __name__ == "__main__":
    main()
