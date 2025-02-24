import os
import librosa
import numpy as np

# Correct dataset path
dataset_path = "C:/Users/ASUS/OneDrive/Desktop/speech_emotion_classification/datasets/hindi"

# Ensure the output directory exists
output_dir = "C:/Users/ASUS/OneDrive/Desktop/speech_emotion_classification/results"
os.makedirs(output_dir, exist_ok=True)

# Define the emotions to process
emotions = ["happy", "neutral", "sad", "fear"]  # Excluding 'angry'

# Initialize feature storage
features = []

# Loop through each emotion folder
for emotion in emotions:
    emotion_path = os.path.join(dataset_path, emotion)

    # Check if folder exists
    if not os.path.exists(emotion_path):
        print(f"⚠️ Skipping missing folder: {emotion_path}")
        continue
    
    print(f"🔍 Processing emotion: {emotion}")

    # Process each file in the folder
    for file in os.listdir(emotion_path):
        if file.endswith(".wav"):
            file_path = os.path.join(emotion_path, file)

            try:
                # Load the audio file
                y, sr = librosa.load(file_path, sr=None)

                # Extract MFCC features
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfccs_mean = np.mean(mfccs, axis=1)  # Take the mean of each MFCC

                # Store the extracted features along with the label
                features.append((mfccs_mean, emotion))

            except Exception as e:
                print(f"❌ Error processing {file}: {e}")

# Convert to numpy array
features = np.array(features, dtype=object)

# Save the extracted features
features_path = os.path.join(output_dir, "features.npy")
np.save(features_path, features)

print(f"✅ Feature extraction complete! Saved to {features_path}")
