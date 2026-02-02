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
    # BODY DETECTORS (for human detection)
    # =========================================================================
    body_detectors = {
        "7": {
            "name": "YOLOv8n (Nano - Fast person detection)",
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
            "convert_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx",
            "filename": "yolov8n.onnx",
            "size": "6.3 MB",
            "gpu": True,
        },
        "8": {
            "name": "YOLOv8s (Small - Better accuracy)",
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.onnx",
            "filename": "yolov8s.onnx",
            "size": "22.5 MB",
            "gpu": True,
        },
    }

    # =========================================================================
    # HUMAN SEGMENTATION (for body contours)
    # =========================================================================
    segmenters = {
        "9": {
            "name": "MediaPipe Selfie Segmentation (Human segmentation)",
            "url": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
            "onnx_url": "https://github.com/PINTO0309/PINTO_model_zoo/raw/main/426_Selfie-Segmentation/selfie_segmentation_landscape.onnx",
            "filename": "human_seg.onnx",
            "size": "0.6 MB",
            "gpu": True,
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

    print("\n--- BODY DETECTORS (Human Detection) ---")
    for key, model in body_detectors.items():
        exists = "✓" if (models_dir / model["filename"]).exists() else " "
        print(f"  [{exists}] {key}. {model['name']}")
        print(f"       GPU | {model['size']}")

    print("\n--- HUMAN SEGMENTATION (Body Contours) ---")
    for key, model in segmenters.items():
        exists = "✓" if (models_dir / model["filename"]).exists() else " "
        print(f"  [{exists}] {key}. {model['name']}")
        print(f"       GPU | {model['size']}")

    print("\n--- OPTIONS ---")
    print("  A. Download all recommended (SCRFD 2.5G + ArcFace R100)")
    print("  B. Download body detection + segmentation (YOLOv8n + PP-HumanSeg)")
    print("  Q. Quit")

    print("\n" + "=" * 70)
    choice = input("Select model to download (1-9, A, B, or Q): ").strip().upper()

    if choice == 'Q':
        print("Cancelled.")
        return

    if choice == 'B':
        # Download body detection and segmentation models
        to_download = []

        # YOLOv8n for body detection
        if not (models_dir / "yolov8n.onnx").exists():
            to_download.append(body_detectors["7"])

        # PP-HumanSeg for segmentation
        if not (models_dir / "human_seg.onnx").exists():
            to_download.append(segmenters["9"])

        if not to_download:
            print("\n✓ Body detection models already exist!")
            return

        for model in to_download:
            output_path = models_dir / model['filename']
            print(f"\nDownloading: {model['name']}")
            # Use onnx_url if available, otherwise use url
            url = model.get('onnx_url', model.get('convert_url', model['url']))
            download_file(url, str(output_path))

        print("\n" + "=" * 70)
        print("✓ Body detection models downloaded!")
        print("\nAdd to config.yaml:")
        print("  detection:")
        print("    body_detection: true")
        print("    segmentation: true")
        print("  models:")
        print("    body_detector: models/yolov8n.onnx")
        print("    human_segmenter: models/human_seg.onnx")
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
    elif choice in body_detectors:
        selected = body_detectors[choice]
    elif choice in segmenters:
        selected = segmenters[choice]
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
