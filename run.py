
import os
import sys
import time
import socket
import subprocess
from pathlib import Path

# Project's main directory
ROOT = Path(__file__).resolve().parent

# Prediction server settings
PREDICTION_HOST = "127.0.0.1"
PREDICTION_PORT = 8765

# Render provides PORT automatically.
# Locally, Streamlit will use port 8501.
STREAMLIT_PORT = int(os.environ.get("PORT", "8501"))

prediction_process = None

try:
    print("Starting emotion prediction server...", flush=True)

    # Start prediction server
    prediction_process = subprocess.Popen(
        [sys.executable, str(ROOT / "prediction_server.py")],
        cwd=ROOT
    )

    # Wait until prediction server is ready
    for attempt in range(30):

        if prediction_process.poll() is not None:
            raise RuntimeError(
                "Prediction server stopped unexpectedly."
            )

        try:
            with socket.create_connection(
                (PREDICTION_HOST, PREDICTION_PORT),
                timeout=1
            ):
                print("Prediction server is ready!", flush=True)
                break

        except OSError:
            time.sleep(1)

    else:
        raise RuntimeError(
            "Prediction server did not start within 30 seconds."
        )

    print("Starting Streamlit application...", flush=True)

    # Start Streamlit
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "app" / "streamlit_app.py"),
            "--server.address=0.0.0.0",
            f"--server.port={STREAMLIT_PORT}",
            "--server.headless=true"
        ],
        cwd=ROOT,
        check=True
    )

finally:
    # Stop prediction server when Streamlit stops
    if prediction_process is not None:
        print("Stopping prediction server...", flush=True)

        prediction_process.terminate()

        try:
            prediction_process.wait(timeout=10)

        except subprocess.TimeoutExpired:
            prediction_process.kill()
            prediction_process.wait()
