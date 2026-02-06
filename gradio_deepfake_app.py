import gradio as gr
import numpy as np
import time
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load your saved model
model = load_model('MODELdeepfakedetector.h5')

# Function to predict if an image is fake or real
def predict_image(img):
    # Start timer
    start_time = time.time()
    
    # Resize and preprocess the image
    img = image.load_img(img, target_size=(128, 128))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img)
    probability = float(prediction[0][0])
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Determine result
    if prediction < 0.5:
        result = "Fake"
    else:
        result = "Real"
    
    # Return all metrics
    return result, f"{probability:.2%}", f"{processing_time:.2f} s"

# Create Gradio interface with multiple outputs
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="filepath"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Probability"),
        gr.Textbox(label="Processing Time")
    ],
    title="Deepfake Detector",
    description="Upload an image to check if it's Real or Fake"
)

# Launch the interface
interface.launch()