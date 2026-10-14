
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

            predictions = model(image, training=False).numpy()[0]
            highest_index = int(np.argmax(predictions))

            # Convert model scores to percentages
            scores = {
                emotion: round(float(score * 100), 2)
                for emotion, score in zip(emotions, predictions)
            }

            result = {
                "emotions": scores,
                "highest_emotion": emotions[highest_index],
                "confidence": round(
                    float(predictions[highest_index] * 100), 2
                )
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
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def log_message(self, format, *args):
        pass


server = HTTPServer(("127.0.0.1", 8765), Handler)
print("Emotion prediction server running on port 8765")
server.serve_forever()

