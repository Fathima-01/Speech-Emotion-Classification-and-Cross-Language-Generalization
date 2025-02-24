import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import pickle

# Define paths
RESULTS_PATH = "../results/"
MODELS_PATH = "../models/"
os.makedirs(MODELS_PATH, exist_ok=True)

# Load features and labels
features_path = os.path.join(RESULTS_PATH, "features.npy")
labels_path = os.path.join(RESULTS_PATH, "labels.npy")

if not os.path.exists(features_path) or not os.path.exists(labels_path):
    raise FileNotFoundError("❌ Feature or label file not found. Ensure feature extraction is complete.")

X = np.load(features_path)
y = np.load(labels_path)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder
with open(os.path.join(RESULTS_PATH, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

# Convert labels to categorical
y_encoded = keras.utils.to_categorical(y_encoded)

# Define a simple CNN model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(40, 128, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation="relu"),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(len(label_encoder.classes_), activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X.reshape(-1, 40, 128, 1), y_encoded, epochs=20, batch_size=32, validation_split=0.2)

# Save trained model
model.save(os.path.join(MODELS_PATH, "emotion_model.h5"))

print("✅ Model training complete. Model saved as /models/emotion_model.h5")
