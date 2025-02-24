import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Path to the features file
FEATURES_PATH = "../results/features.npy"

# Load features and labels
data = np.load(FEATURES_PATH, allow_pickle=True)
X = np.array([item[0] for item in data])  # Feature vectors
y = np.array([item[1] for item in data])  # Labels (emotions)

# Encode labels into numeric format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple MLP model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(set(y)), activation='softmax')  # Output layer with softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model performance
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Save the trained model
MODEL_PATH = "../models/emotion_model.h5"
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")

# Save label encoder for inference
np.save("../models/label_encoder.npy", label_encoder.classes_)
print("✅ Label encoder saved.")
