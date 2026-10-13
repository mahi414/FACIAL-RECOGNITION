from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import base64
import numpy as np
from tensorflow.keras.models import load_model

emotions = [
    "Angry", "Disgust", "Fear", "Happy",
    "Neutral", "Sad", "Surprise"
]

model = load_model("models/emotion_model.keras")

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(length))

            raw = base64.b64decode(body["image"])
            image = np.frombuffer(raw, dtype=np.uint8)
            image = image.reshape(1, 48, 48, 1).astype(np.float32) / 255.0

            prediction = model(image, training=False).numpy()[0]
            index = int(np.argmax(prediction))

            result = {
                "emotion": emotions[index],
                "confidence": float(prediction[index] * 100)
            }

            data = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        except Exception as error:
            data = json.dumps({"error": str(error)}).encode()
            self.send_response(400)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def log_message(self, format, *args):
        pass

server = HTTPServer(("127.0.0.1", 8765), Handler)
print("Emotion prediction server running on port 8765")
server.serve_forever()
