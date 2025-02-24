import os
import numpy as np
import librosa
import librosa.display
import glob
import pandas as pd

# Define paths
DATASET_PATH = "../datasets/"
RESULTS_PATH = "../results/"

# Ensure results directory exists
os.makedirs(RESULTS_PATH, exist_ok=True)

# Function to extract MFCC features
def extract_features(file_path, max_pad_length=128):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    
    # Padding or truncating to ensure fixed size
    pad_width = max_pad_length - mfcc.shape[1]
    if pad_width > 0:
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_length]
    
    return mfcc

# Load dataset (Modify this based on your dataset structure)
audio_files = glob.glob(DATASET_PATH + "**/*.wav", recursive=True)
labels = []

features = []
for file in audio_files:
    label = os.path.basename(os.path.dirname(file))  # Assumes folder name is the label
    labels.append(label)
    features.append(extract_features(file))

features = np.array(features)
labels = np.array(labels)

# Save extracted features and labels
np.save(os.path.join(RESULTS_PATH, "features.npy"), features)
np.save(os.path.join(RESULTS_PATH, "labels.npy"), labels)

print("✅ Feature extraction completed. Files saved in /results/")
print(f"🔹 Features shape: {features.shape}")
print(f"🔹 Labels shape: {labels.shape}")
