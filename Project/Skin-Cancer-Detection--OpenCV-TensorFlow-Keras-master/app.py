'''import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('skin_cancer_detection_model.h5')

# Define the size of the input images
img_size = (224, 224)

# Define a function to preprocess the input image
def preprocess_image(img):
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Define the Streamlit app
def app():
    st.title('Skin Cancer Detection App')

    # Allow the user to upload an image
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        # Read the image
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        # Display the image

        st.image(img, caption='Uploaded Image', use_column_width=True)
        # Preprocess the image
        img = preprocess_image(img)

        # Make a prediction
        pred = model.predict(img)
        pred_label = 'Cancer' if pred[0][0] > 0.5 else 'Not Cancer'
        pred_prob = pred[0][0]
        
        # Show the prediction result
        st.write(f'Prediction: {pred_label}')
        st.write(f'Probability Of Skin Cancer: {pred_prob:.2f}')

# Run the app
if __name__ == '__main__':
    app()'''
import streamlit as st
import os
from PIL import Image
import cv2
import numpy as np
from model import img_size

import shutil

# Landing Page Introduction
st.title("Easy Skin Health Checker")
st.markdown("""
    Welcome to the Easy Skin Health Checker!
    This app helps you perform a quick check of skin moles to provide preliminary guidance.
    Please note that the results are not a substitute for professional medical advice. Your privacy is our priority, and images are deleted immediately after processing.
    """)

# Navigation Tabs
tabs = st.tabs(["Upload Image", "Detection Results", "Disclaimer"])

# Image Upload Section
with tabs[0]:
    st.header("Step 1: Upload an Image of Your Mole")
    st.text("You can either drag and drop or click the button to upload.")
    uploaded_image = st.file_uploader("Upload an image of your skin mole", type=["jpg", "jpeg", "png"])

    # Display Thumbnail of Uploaded Image
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.text("Processing...")
        # Save image temporarily
        image_path = f"temp_{uploaded_image.name}"
        image.save(image_path)

# Define a function to preprocess the input image
def preprocess_image(img):
    img = cv2.resize(img, img_size)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# Results Translation Section
with tabs[1]:
    if uploaded_image is not None:
        st.header("Step 2: Detection Results")
        # Mock result for demonstration purposes
        import random

        result = random.choice(["Healthy", "Check with a doctor", "Urgent Attention Required"])
        confidence = random.uniform(0.7, 1.0)

        # Simplified Output Display
        if result == "Healthy":
            st.markdown("<h3 style='color:green;'>The mole looks healthy!</h3>", unsafe_allow_html=True)
        elif result == "Check with a doctor":
            st.markdown("<h3 style='color:yellow;'>This mole might require further medical attention.</h3>",
                        unsafe_allow_html=True)
        else:
            st.markdown(
                "<h3 style='color:red;'>Urgent Attention Required: Please consult a doctor as soon as possible.</h3>",
                unsafe_allow_html=True)

        # Confidence Level
        st.progress(confidence)
        st.write(f"Model Confidence: {confidence * 100:.2f}%")

        # Delete the image after processing
        if os.path.exists(image_path):
            os.remove(image_path)

# Disclaimer Section
with tabs[2]:
    st.header("Disclaimer")
    st.markdown("""
        This app is intended to provide preliminary guidance only and should not be used as a substitute for a professional medical diagnosis.
        Please consult a healthcare provider for any concerns about your skin health.
    """)
    agree = st.checkbox("I understand and accept the disclaimer.")

    if not agree:
        st.warning("You must accept the disclaimer before using the app.")

# Help Section
st.sidebar.header("Help & FAQ")
st.sidebar.markdown("""
    **Q: How do I take a good picture of my mole?**
    - Make sure the mole is well-lit and centered in the photo.

    **Q: Is this app a substitute for a dermatologist?**
    - No, this app is for informational purposes only. Please consult a healthcare provider for medical advice.
""")

st.image(img, caption='Uploaded Image', use_column_width=True)
# Preprocess the image
img = preprocess_image(img)

# Make a prediction
pred = model.predict(img)
pred_label = 'Cancer' if pred[0][0] > 0.5 else 'Not Cancer'
pred_prob = pred[0][0]

# Show the prediction result
st.write(f'Prediction: {pred_label}')
st.write(f'Probability Of Skin Cancer: {pred_prob:.2f}')
# Aesthetic Improvements
st.sidebar.header("Settings")
light_dark_mode = st.sidebar.radio("Choose Display Mode", ("Light Mode", "Dark Mode"))

if light_dark_mode == "Dark Mode":
    st.markdown("<style>body { background-color: #2c2c2c; color: #ffffff; }</style>", unsafe_allow_html=True)

# Footer
st.sidebar.header("Contact Us")
st.sidebar.info("For non-medical support, please contact support@skincancerchecker.com.")