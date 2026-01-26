# Raspberry Pi Deployment

## Prerequisites

- Raspberry Pi 4 (4GB+ recommended)
- Raspberry Pi Camera Module or USB camera
- Raspberry Pi OS (64-bit recommended)

## Installation

```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-opencv python3-picamera2

# Clone repository
git clone <repo-url> ~/WatchBird && cd ~/WatchBird

# Create venv with system packages (required for picamera2)
uv venv --system-site-packages
uv pip install -e ".[rpi,cpu,dev]"


# Download models
python tools/download_models.py
```

## Enroll & Run

```bash
# Enroll a person
python tools/auto_enroll.py --person yourname --backend picamera2 --auto-enroll

# Run recognition
python tools/run_runtime.py --backend picamera2
```

View stream: `http://<pi-ip>:8080/stream`

## Auto-Start (systemd)

```bash
sudo tee /etc/systemd/system/watchbird.service << EOF
[Unit]
Description=WatchBird Recognition
After=network.target

[Service]
User=$USER
WorkingDirectory=$HOME/WatchBird
ExecStart=$HOME/WatchBird/.venv/bin/python tools/run_runtime.py --backend picamera2
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable watchbird
sudo systemctl start watchbird
```

## Performance Tips

- Use `mobilefacenet.onnx` for faster inference
- Set `embedding_sample_interval: 5` in config.yaml
- Lower resolution to 480x360 if needed
