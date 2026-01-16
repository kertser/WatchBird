# Raspberry Pi 4 Deployment Checklist

Complete checklist for deploying WatchBird on Raspberry Pi 4.

## ✅ Pre-Deployment Checklist

### Hardware
- [ ] Raspberry Pi 4 (4GB RAM minimum, 8GB recommended)
- [ ] MicroSD card (32GB+ recommended)
- [ ] Power supply (official 5V 3A recommended)
- [ ] CSI camera module (optional, can test with video files first)
- [ ] Network connection (WiFi or Ethernet)
- [ ] Case with cooling (heatsinks/fan recommended)

### Software Prerequisites
- [ ] Raspberry Pi OS (64-bit recommended)
- [ ] SSH enabled
- [ ] Python 3.11+ installed
- [ ] Internet connection for package downloads

## 📦 Installation Steps

### 1. System Setup

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    git \
    cmake \
    build-essential \
    libopencv-dev \
    libatlas-base-dev \
    libhdf5-dev

# Install Picamera2 (if using CSI camera)
sudo apt-get install -y python3-picamera2
```

### 2. Clone Repository

```bash
cd ~
git clone <your-repo-url> WatchBird
cd WatchBird
```

### 3. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel
```

### 4. Install Python Dependencies

```bash
# Install core dependencies first
python -m pip install numpy

# Install OpenCV (headless version for no GUI)
python -m pip install opencv-python-headless

# Install ONNX Runtime (CPU-only for RPi)
python -m pip install onnxruntime

# Install FAISS (CPU version)
python -m pip install faiss-cpu

# Install remaining dependencies
python -m pip install pyyaml flask

# Install package in editable mode
python -m pip install -e .
```

### 5. Verify Installation

```bash
python tools/test_installation.py
```

Expected output: All tests should pass ✅

## 🤖 Model Setup

### Download Face Embedder

```bash
# Create models directory
mkdir -p models

# Download ArcFace model (or use your own)
wget https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx \
     -O models/mobilefacenet.onnx

# Verify model exists
ls -lh models/mobilefacenet.onnx
```

### Optional: Download Person Detector

```bash
# For future multi-modal use
python -m pip install ultralytics
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='onnx')"
mv yolov8n.onnx models/yolo_nano.onnx
```

## 📸 Camera Setup (Optional)

### Enable CSI Camera

```bash
# Check if camera is detected
libcamera-hello --list-cameras

# Test camera capture
libcamera-jpeg -o test.jpg
```

### Configure Picamera2

Edit `config.yaml`:
```yaml
camera:
  backend: picamera2
  resolution: [640, 480]  # Adjust as needed
  fps: 15                  # Lower for better stability
```

## 👥 Enrollment

### 1. Prepare Identity Images

```bash
mkdir -p friendly

# Create folders for each person
mkdir -p friendly/person1 friendly/person2

# Copy images (5-10 per person recommended)
# Transfer from another computer via SCP:
scp -r /local/path/to/images/* pi@<rpi-ip>:~/WatchBird/friendly/
```

### 2. Run Enrollment

```bash
source .venv/bin/activate
cd ~/WatchBird

python tools/enroll.py \
    --data-dir friendly \
    --config config.yaml \
    --min-quality 0.4
```

### 3. Verify Index Created

```bash
ls -lh data/index/
# Should see: face.index and meta.jsonl
```

## 🚀 Runtime Deployment

### Test with Video File First

```bash
# Copy test video to RPi
scp test_video.mp4 pi@<rpi-ip>:~/WatchBird/

# Run runtime
python tools/run_runtime.py \
    --backend video_file \
    --video test_video.mp4 \
    --config config.yaml
```

### Run with Live Camera

```bash
python tools/run_runtime.py \
    --backend picamera2 \
    --config config.yaml
```

### View Debug Stream

From your computer, open browser:
```
http://<rpi-ip>:8080/stream
```

## 🔧 Performance Tuning

### Optimize for RPi4

Edit `config.yaml`:

```yaml
# Lower resolution for better performance
camera:
  resolution: [320, 240]  # Start small
  fps: 10                  # Lower FPS = less CPU

# Adjust detection thresholds
detection:
  face_conf_threshold: 0.75  # Higher = fewer detections

# Increase window size for more stable decisions
fusion:
  window_size: 15
  consistency_count: 10
```

### Monitor Performance

```bash
# Terminal 1: Run runtime
python tools/run_runtime.py --backend picamera2 --config config.yaml

# Terminal 2: Monitor CPU/memory
htop
# or
watch -n 1 'vcgencmd measure_temp && free -h'
```

### Expected Performance
- **FPS**: 5-10 fps (acceptable)
- **CPU**: 60-80% on 1-2 cores
- **RAM**: 500-800 MB
- **Temperature**: <70°C with cooling

## 🔒 Security Hardening

### Restrict MJPEG Server

Edit `config.yaml`:
```yaml
stream:
  enabled: true
  host: 127.0.0.1  # Localhost only
  port: 8080
```

Then use SSH tunnel:
```bash
ssh -L 8080:localhost:8080 pi@<rpi-ip>
# Access on your computer: http://localhost:8080/stream
```

### Firewall Rules

```bash
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8080/tcp  # MJPEG (if needed remotely)
sudo ufw enable
```

## 🔄 Auto-Start on Boot

### Create systemd Service

```bash
sudo nano /etc/systemd/system/WatchBird.service
```

Add:
```ini
[Unit]
Description=WatchBird Recognition Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/WatchBird
Environment="PATH=/home/pi/WatchBird/.venv/bin"
ExecStart=/home/pi/WatchBird/.venv/bin/python tools/run_runtime.py --backend picamera2 --config config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable WatchBird
sudo systemctl start WatchBird

# Check status
sudo systemctl status WatchBird

# View logs
sudo journalctl -u WatchBird -f
```

## 📊 Monitoring & Logging

### Log to File

Modify systemd service:
```ini
StandardOutput=append:/var/log/WatchBird/output.log
StandardError=append:/var/log/WatchBird/error.log
```

Create log directory:
```bash
sudo mkdir -p /var/log/WatchBird
sudo chown pi:pi /var/log/WatchBird
```

### Event Logging

Redirect events to file:
```bash
python tools/run_runtime.py \
    --backend picamera2 \
    --config config.yaml \
    > /var/log/WatchBird/events.jsonl 2>&1
```

## 🐛 Troubleshooting

### Low FPS

1. Reduce resolution: `[320, 240]` or `[160, 120]`
2. Lower FPS target: `fps: 5`
3. Skip frames in detection
4. Use lighter models

### High CPU Temperature

1. Add heatsinks/fan
2. Reduce FPS
3. Lower resolution
4. Disable MJPEG streaming when not needed

### Memory Issues

1. Close other applications
2. Use swap if needed:
   ```bash
   sudo dphys-swapfile swapoff
   sudo nano /etc/dphys-swapfile
   # Set CONF_SWAPSIZE=2048
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

### Picamera2 Errors

1. Check camera connection
2. Update firmware: `sudo rpi-update`
3. Verify libcamera: `libcamera-hello`
4. Use video file backend for testing

### FAISS/ONNX Installation Issues

On ARM64:
```bash
# Install from conda-forge
python -m pip install --extra-index-url https://www.piwheels.org/simple/ faiss-cpu
```

## ✅ Deployment Verification

### Final Checklist

- [ ] System updated and dependencies installed
- [ ] Virtual environment created and activated
- [ ] Python packages installed (`test_installation.py` passes)
- [ ] Models downloaded and placed in `models/`
- [ ] Configuration file customized for your setup
- [ ] Enrollment completed with test identities
- [ ] FAISS index created in `data/index/`
- [ ] Runtime tested with video file
- [ ] Runtime tested with live camera (if applicable)
- [ ] MJPEG stream accessible from browser
- [ ] Performance metrics acceptable (5+ FPS)
- [ ] Temperature under control (<70°C)
- [ ] Auto-start service configured (optional)
- [ ] Logs being captured properly

## 📞 Support

If issues persist:

1. Check logs: `sudo journalctl -u WatchBird -n 100`
2. Test components individually
3. Verify model compatibility
4. Check network connectivity
5. Review configuration file

## 🎉 Deployment Complete!

Once all checklist items are complete, your WatchBird system is ready for production use on Raspberry Pi 4!


