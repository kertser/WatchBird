# Deployment

## Raspberry Pi 4

### Requirements
- Raspberry Pi 4 (4GB+ RAM recommended)
- Raspberry Pi Camera Module v2 or USB webcam
- Python 3.9+

### Install

```bash
# System dependencies
sudo apt update
sudo apt install -y python3-pip python3-venv libopencv-dev

# Clone and setup
git clone <repo> && cd WatchBird
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Download models
python tools/download_models.py
```

### Camera Setup

**Pi Camera:**
```bash
# Enable camera in raspi-config
sudo raspi-config  # Interface Options → Camera → Enable

# Install picamera2
pip install picamera2
```

**USB Camera:**
```bash
# Check if detected
ls /dev/video*
```

### Configuration for Pi

```yaml
# config.yaml optimized for Pi 4
camera:
  backend: picamera2        # or usb
  resolution: [640, 480]    # Lower for speed
  fps: 15                   # Realistic for Pi

models:
  face_embedder: models/mobilefacenet.onnx  # Fastest

inference:
  use_gpu: false            # Pi has no GPU

fusion:
  embedding_sample_interval: 3  # Less frequent
  embedding_min: 4              # Faster decisions
```

### Run as Service

```bash
# Create service file
sudo nano /etc/systemd/system/watchbird.service
```

```ini
[Unit]
Description=WatchBird Face Recognition
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/WatchBird
Environment="PATH=/home/pi/WatchBird/.venv/bin"
ExecStart=/home/pi/WatchBird/.venv/bin/python tools/run_runtime.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable watchbird
sudo systemctl start watchbird

# Check status
sudo systemctl status watchbird
```

### Performance Tips

1. **Use MobileFaceNet** - 10x faster than ArcFace
2. **Lower resolution** - 640x480 is sufficient
3. **Reduce FPS** - 15fps is enough for recognition
4. **Sample embeddings** - Set `embedding_sample_interval: 3`
5. **Overclock** (optional) - Can improve ~20%

Expected performance on Pi 4:
- Detection: ~20ms/frame
- Embedding: ~50ms/face
- Overall: ~5-8 FPS

---

## Windows / Linux Desktop

### Requirements
- Python 3.9+
- NVIDIA GPU (optional, for speed)

### GPU Acceleration

**NVIDIA (CUDA):**
```bash
pip install onnxruntime-gpu
```

**AMD/Intel (DirectML - Windows only):**
```bash
pip install onnxruntime-directml
```

### Run

```bash
python tools/run_runtime.py --backend usb
```

Expected performance with GPU:
- Overall: 15-30 FPS

---

## Docker (Experimental)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -e .
RUN python tools/download_models.py

EXPOSE 8080

CMD ["python", "tools/run_runtime.py"]
```

```bash
docker build -t watchbird .
docker run -p 8080:8080 --device=/dev/video0 watchbird
```
