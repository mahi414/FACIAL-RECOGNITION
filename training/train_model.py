import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

TRAIN_PATH = "dataset/train"
MODEL_PATH = "models/emotion_model_candidate.keras"

emotions = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise"
]

# 1. Load dataset
X, y = [], []

for label, emotion in enumerate(emotions):
    folder = os.path.join(TRAIN_PATH, emotion)

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Missing dataset folder: {folder}")

    count = 0
    for filename in sorted(os.listdir(folder)):
        image_path = os.path.join(folder, filename)

        try:
            with Image.open(image_path) as img:
                image = np.array(img.convert("L").resize((48, 48)))
            X.append(image)
            y.append(label)
            count += 1
        except (OSError, ValueError):
            print(f"Skipping unreadable file: {image_path}")

    print(f"{emotion}: {count} images")

X = np.asarray(X, dtype=np.float32)[..., np.newaxis] / 255.0
y = np.asarray(y, dtype=np.int64)

print("Total images:", len(X))

# 2. Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)

# 3. Balance classes during training
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weights = dict(zip(classes.tolist(), weights.tolist()))
print("Class weights:", class_weights)

# 4. Data augmentation (training images only)
augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomZoom(0.10),
], name="data_augmentation")

# 5. Improved CNN
inputs = layers.Input(shape=(48, 48, 1))
x = augmentation(inputs)

for filters in [32, 64, 128]:
    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.20)(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.40)(x)
outputs = layers.Dense(7, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 6. Save the best candidate; preserve the current production model
os.makedirs("models", exist_ok=True)

training_callbacks = [
    callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    class_weight=class_weights,
    callbacks=training_callbacks
)

# 7. Report best validation result
best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1
best_accuracy = max(history.history["val_accuracy"])

print(f"\nBest validation accuracy: {best_accuracy:.4f}")
print(f"Best epoch: {best_epoch}")
print(f"Candidate saved at: {MODEL_PATH}")
print("Existing emotion_model.keras has NOT been replaced.")
