"""Download face detection and recognition models."""
import os
import sys
import urllib.request
from pathlib import Path

def download_file(url: str, output_path: str) -> bool:
    """Download file with progress bar."""
    try:
        print(f"Downloading from: {url}")
        print(f"Saving to: {output_path}")

        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                bar_length = 50
                filled = int(bar_length * percent / 100)
                bar = '█' * filled + '-' * (bar_length - filled)
                print(f'\r[{bar}] {percent:.1f}%', end='')

        urllib.request.urlretrieve(url, output_path, show_progress)
        print("\n✓ Download complete!")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False

def main():
    """Download face detection and recognition models."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("WatchBird - Model Downloader")
    print("=" * 70)

    # =========================================================================
    # DETECTORS
    # =========================================================================
    detectors = {
        "1": {
            "name": "SCRFD 2.5G (Recommended - GPU + Landmarks)",
            "url": "https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/det_2.5g.onnx",
            "filename": "scrfd_2.5g.onnx",
            "size": "3.1 MB",
            "gpu": True,
            "landmarks": True,
        },
        "2": {
            "name": "SCRFD 10G (Most Accurate - GPU + Landmarks)",
            "url": "https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/det_10g.onnx",
            "filename": "scrfd_10g.onnx",
            "size": "16.1 MB",
            "gpu": True,
            "landmarks": True,
        },
        "3": {
            "name": "SCRFD 500M (Fastest - GPU + Landmarks)",
            "url": "https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/det_500m.onnx",
            "filename": "scrfd_500m.onnx",
            "size": "2.4 MB",
            "gpu": True,
            "landmarks": True,
        },
        "4": {
            "name": "UltraFace RFB-320 (Fast - GPU, No Landmarks)",
            "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/ultraface/models/version-RFB-320.onnx",
            "filename": "ultraface_rfb320.onnx",
            "size": "1.2 MB",
            "gpu": True,
            "landmarks": False,
        },
    }

    # =========================================================================
    # EMBEDDERS
    # =========================================================================
    embedders = {
        "1": {
            "name": "ArcFace R100 (Best Accuracy)",
            "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
            "filename": "arcface_r100.onnx",
            "size": "249 MB",
        },
        "2": {
            "name": "MobileFaceNet (Fast, Good Accuracy)",
            "url": "https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/w600k_r50.onnx",
            "filename": "mobilefacenet.onnx",
            "size": "166 MB",
        },
    }

    # =========================================================================
    # MENU
    # =========================================================================
    print("\n--- FACE DETECTORS ---")
    for key, model in detectors.items():
        gpu_str = "GPU" if model["gpu"] else "CPU"
        lm_str = "✓ Landmarks" if model["landmarks"] else "✗ No Landmarks"
        exists = "✓" if (models_dir / model["filename"]).exists() else " "
        print(f"  [{exists}] {key}. {model['name']}")
        print(f"       {gpu_str} | {lm_str} | {model['size']}")

    print("\n--- FACE EMBEDDERS ---")
    for key, model in embedders.items():
        exists = "✓" if (models_dir / model["filename"]).exists() else " "
        print(f"  [{exists}] {int(key)+4}. {model['name']}")
        print(f"       {model['size']}")

    print("\n--- OPTIONS ---")
    print("  A. Download all recommended (SCRFD 2.5G + ArcFace R100)")
    print("  Q. Quit")

    print("\n" + "=" * 70)
    choice = input("Select model to download (1-6, A, or Q): ").strip().upper()

    if choice == 'Q':
        print("Cancelled.")
        return

    if choice == 'A':
        # Download recommended set
        to_download = [
            detectors["1"],  # SCRFD 2.5G
        ]
        # Check if arcface exists
        if not (models_dir / "arcface_r100.onnx").exists():
            to_download.append(embedders["1"])  # ArcFace R100

        for model in to_download:
            output_path = models_dir / model['filename']
            if output_path.exists():
                print(f"\n✓ {model['filename']} already exists, skipping...")
                continue
            print(f"\nDownloading: {model['name']}")
            download_file(model['url'], str(output_path))

        print("\n" + "=" * 70)
        print("✓ Recommended models downloaded!")
        print("\nUpdate config.yaml:")
        print("  detection:")
        print("    detector_type: scrfd")
        print("  models:")
        print("    face_detector: models/scrfd_2.5g.onnx")
        print("    face_embedder: models/arcface_r100.onnx")
        return

    # Map choice to model
    if choice in detectors:
        selected = detectors[choice]
    elif choice in ['5', '6']:
        selected = embedders[str(int(choice) - 4)]
    else:
        print("Invalid choice.")
        return

    output_path = models_dir / selected['filename']

    if output_path.exists():
        overwrite = input(f"\n{output_path} already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Cancelled.")
            return

    print(f"\nDownloading: {selected['name']}")
    print(f"Expected size: {selected['size']}")
    print()

    if download_file(selected['url'], str(output_path)):
        print(f"\n✓ Model saved to: {output_path}")
    else:
        print("\n✗ Failed to download model")
        print("\nManual download:")
        print(f"URL: {selected['url']}")
        print(f"Save to: {output_path}")

if __name__ == "__main__":
    main()
