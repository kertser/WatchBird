# Raspberry Pi Deployment

## Install

```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-opencv python3-picamera2

# Clone and setup
git clone <repo-url> ~/WatchBird && cd ~/WatchBird

# Using uv (recommended)
rm -rf .venv
uv venv --system-site-packages
uv sync

# Or using pip
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Download Models

```bash
python tools/download_models.py
```

## Enroll & Run

```bash
# Enroll person
uv run python tools/capture_and_enroll.py --person mike --backend picamera2 --count 10 --auto-enroll

# Run recognition
uv run python tools/run_runtime.py --backend picamera2 --config config.yaml
```

## Auto-Start (systemd)

```bash
# Create service file
sudo tee /etc/systemd/system/watchbird.service << EOF
[Unit]
Description=WatchBird Recognition
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/WatchBird
ExecStart=/home/pi/WatchBird/.venv/bin/python tools/run_runtime.py --backend picamera2
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl enable watchbird
sudo systemctl start watchbird
```

## View Stream

Open browser: `http://<pi-ip>:8080/stream`
