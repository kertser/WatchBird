from picamera2 import Picamera2
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading, time
import cv2

latest_jpeg = None
lock = threading.Lock()

def capture_loop():
    global latest_jpeg

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_video_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
    )
    picam2.start()
    time.sleep(0.2)

    while True:
        frame = picam2.capture_array()   # numpy RGB image

        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        with lock:
            latest_jpeg = jpeg.tobytes()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/stream"):
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        while True:
            with lock:
                frame = latest_jpeg

            if frame is None:
                time.sleep(0.05)
                continue

            self.wfile.write(b"--frame\r\n")
            self.wfile.write(b"Content-Type: image/jpeg\r\n")
            self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
            self.wfile.write(frame)
            self.wfile.write(b"\r\n")
            time.sleep(0.05)  # ~20 FPS

if __name__ == "__main__":
    threading.Thread(target=capture_loop, daemon=True).start()
    server = HTTPServer(("0.0.0.0", 8080), Handler)
    print("MJPEG stream: http://<pi-ip>:8080/stream")
    server.serve_forever()
