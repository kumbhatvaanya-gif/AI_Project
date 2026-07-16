import tensorflow as tf

# Define the exact dimensions your Streamlit app expects
IMG_SIZE = (128, 128, 3)
NUM_CLASSES = 5

print("🏗️ Building dummy model architecture...")
dummy_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMG_SIZE),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

print("⚡ Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(dummy_model)
tflite_dummy = converter.convert()

# Save it to your computer
file_name = "model.tflite"
with open(file_name, "wb") as f:
    f.write(tflite_dummy)

print(f"✅ Success! '{file_name}' has been generated.")
