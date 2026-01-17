rm -rf .venv
uv venv --system-site-packages
uv sync
uv run python tools/capture_and_enroll.py --person mike --backend picamera2 --count 10 --auto-enroll
uv run python tools/run_runtime.py --backend picamera2 --config config.yaml