"""Download better face recognition models."""
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
            percent = min(100, downloaded * 100 / total_size)
            bar_length = 50
            filled = int(bar_length * downloaded / total_size)
            bar = '█' * filled + '-' * (bar_length - filled)
            print(f'\r[{bar}] {percent:.1f}% ({downloaded}/{total_size} bytes)', end='')

        urllib.request.urlretrieve(url, output_path, show_progress)
        print("\n✓ Download complete!")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False

def main():
    """Download recommended face recognition models."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("WatchBird - Better Face Recognition Model Downloader")
    print("=" * 70)

    models = {
        "1": {
            "name": "ArcFace ResNet100 (Highest Accuracy, Slower)",
            "url": "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
            "filename": "arcface_r100.onnx",
            "size": "~250 MB",
            "accuracy": "99.8%",
            "speed": "Slow (~500ms on RPi4)"
        },
        "2": {
            "name": "MobileFaceNet v2 (Balanced, Recommended)",
            "url": "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
            "filename": "mobilefacenet_v2.onnx",
            "size": "~4 MB",
            "accuracy": "99.5%",
            "speed": "Fast (~100ms on RPi4)"
        },
        "3": {
            "name": "SFace (Sigmoid-Constrained, Robust)",
            "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
            "filename": "sface.onnx",
            "size": "~43 MB",
            "accuracy": "99.6%",
            "speed": "Medium (~200ms on RPi4)"
        }
    }

    print("\nAvailable models:")
    for key, model in models.items():
        print(f"\n{key}. {model['name']}")
        print(f"   Accuracy: {model['accuracy']}")
        print(f"   Speed: {model['speed']}")
        print(f"   Size: {model['size']}")

    print("\n" + "=" * 70)
    choice = input("Select model to download (1-3, or 'q' to quit): ").strip()

    if choice.lower() == 'q':
        print("Cancelled.")
        return

    if choice not in models:
        print("Invalid choice.")
        return

    selected = models[choice]
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
        print("\nNext steps:")
        print(f"1. Update config.yaml:")
        print(f"   models:")
        print(f"     face_embedder: {output_path}")
        print(f"\n2. Re-run enrollment:")
        print(f"   python tools/enroll.py --data-dir friendly --config config.yaml")
        print(f"\n3. Test recognition:")
        print(f"   python tools/run_runtime.py --backend usb --config config.yaml")
    else:
        print("\n✗ Failed to download model")
        print("\nManual download:")
        print(f"URL: {selected['url']}")
        print(f"Save to: {output_path}")

if __name__ == "__main__":
    main()
