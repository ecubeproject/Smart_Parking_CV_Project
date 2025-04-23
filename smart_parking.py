import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # Use CPU only for running app

# Define token extractor in case it was used as a Lambda during model definition
def extract_token(x):
    return x[:, 0]

# Load the model safely with custom_objects fallback
model = load_model(
    "vit_model_savedmodel",
    custom_objects={"extract_token": tf.keras.layers.Lambda(extract_token, name="ExtractToken")}
)

# Inference function
def predict_parking(image):
    if image is None:
        return "Upload an image first", ""
    
    HEIGHT, WIDTH = 224, 224

    # Resize and normalize
    image_resized = cv2.resize(image, (WIDTH, HEIGHT))
    image_input = np.expand_dims(image_resized, axis=0).astype(np.float32) / 255.0

    # Predict
    prob = model.predict(image_input, verbose=0).squeeze()

    # Interpret result
    prediction = "occupied" if prob > 0.57 else "empty"

    return f"{prob:.4f}", prediction

# Gradio interface setup
interface = gr.Interface(
    fn=predict_parking,
    inputs=gr.Image(type="numpy", label="Upload Parking Lot Image"),
    outputs=[
        gr.Textbox(label="Probability (Occupied)"),
        gr.Textbox(label="Prediction @ Threshold 0.57")
    ],
    title="Smart Parking: Detect Parking Lot Occupancy",
    description="Upload an image of a parking lot. The model will predict whether it's occupied or empty based on a threshold of 0.57."
)

# Launch the app
interface.launch()
