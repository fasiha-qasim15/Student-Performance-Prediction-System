
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("dataset/student_performance.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

# Input features
X = data[
    [
        "attendance",
        "study_hours",
        "previous_marks",
        "assignment_marks",
        "quiz_marks"
    ]
].values.astype("float32")

# Target
y = data["performance"].values.astype("int32")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Normalize features
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(X_train)

# Build neural network
model = tf.keras.Sequential([
    tf.keras.Input(shape=(5,)),
    normalizer,
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
print("\nStarting model training...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.20,
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\nModel Evaluation")
print("Test Loss:", loss)
print("Test Accuracy:", accuracy * 100, "%")

# Predictions
predictions = model.predict(X_test, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    predicted_classes,
    target_names=["Low", "Medium", "High"],
    zero_division=0
))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predicted_classes))

# Save trained model
os.makedirs("model", exist_ok=True)

model.save("model/student_performance_model.keras")

print("\nModel training completed successfully!")
print("Model saved in model folder.")
