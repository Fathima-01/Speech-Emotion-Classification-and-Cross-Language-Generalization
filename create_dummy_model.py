from sklearn.svm import SVC
import numpy as np
import joblib

# Dummy training data (just for testing)
X_train = np.random.rand(100, 13)  # 100 samples, 13 MFCC features
y_train = np.random.choice(["happy", "sad", "angry", "neutral", "fear"], 100)

# Train a simple classifier
model = SVC(probability=True)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "models/pretrained_model.pkl")
print("✅ Dummy model saved as models/pretrained_model.pkl")
