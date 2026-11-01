# Real-Time Facial Emotion Detection

A Python-based facial emotion recognition application that detects faces and predicts emotions using a trained deep learning model. The application provides a Streamlit interface for webcam-based detection and image uploads.

## Features

- Real-time facial emotion detection using a webcam
- Upload an image to predict facial emotions
- Face detection using OpenCV
- Deep learning-based emotion classification
- Displays scores for seven emotion classes
- Streamlit web interface
- Separate prediction server for model inference

## Emotions Detected

The model classifies facial expressions into seven categories:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pillow
- Streamlit

## Project Structure

```text
Facial Recognition/
├── app/
│   └── streamlit_app.py
├── models/
│   └── emotion_model.keras
├── training/
│   ├── train_model.py
│   ├── test_model.py
│   ├── check_dataset.py
│   ├── load_dataset.py
│   ├── visualize.py
│   └── predict_image.py
├── dataset/
├── prediction_server.py
├── README.md
└── requirements.txt
```

*The dataset and virtual environment may be excluded from version control.*

## Prerequisites

- Python 3.10 or a compatible Python version
- pip
- Webcam (optional, for live detection)

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/mahi414/FACIAL-RECOGNITION.git
cd FACIAL-RECOGNITION
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Ensure the trained model exists at:

```text
models/emotion_model.keras
```

## Run the Application

### 1. Start the prediction server

Open a terminal and run:

```bash
python prediction_server.py
```

Keep this terminal running.

### 2. Start the Streamlit interface

Open a second terminal, activate the same virtual environment, and run:

```bash
streamlit run app/streamlit_app.py
```

Open the local URL displayed by Streamlit, usually:

```text
http://localhost:8501
```

## Train the Model

Ensure the training dataset is arranged by emotion class under `dataset/train/`, then run:

```bash
python training/train_model.py
```

Training may take time depending on your hardware. Follow the training script's output for the saved model path.

## Evaluate the Model

To evaluate the active model on the test dataset:

```bash
python training/test_model.py
```

The test dataset should be arranged under `dataset/test/` using the same seven emotion folder names.

## Model Performance

The current model achieved approximately **54.27% test accuracy** on the available test dataset during evaluation. Performance may vary on real-world webcam images.

## Limitations

- Predictions can be affected by lighting, camera quality, head pose, and image resolution.
- Emotion predictions are estimates of facial expressions and do not reliably reveal a person's actual feelings.
- The model may perform differently across people and environments.

## Future Improvements

- Improve classification accuracy
- Analyze per-emotion performance with a confusion matrix
- Improve face alignment and preprocessing
- Evaluate performance under different lighting conditions

## Author

**Mahi Jha**

GitHub: [mahi414](https://github.com/mahi414)

---

This project is intended for educational and research purposes.
