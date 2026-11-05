
import tensorflow as tf
import os

# Load trained model
model = tf.keras.models.load_model(
    "model/student_performance_model.keras"
)

print("Trained model loaded successfully!")

# Convert model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

# Create output folder
os.makedirs("tflite_model", exist_ok=True)

# Save TFLite model
with open(
    "tflite_model/student_performance.tflite",
    "wb"
) as file:
    file.write(tflite_model)

print("TensorFlow Lite conversion successful!")
print("TFLite model saved successfully!")
