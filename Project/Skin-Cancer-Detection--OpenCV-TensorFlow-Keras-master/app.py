import streamlit as st
import os
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Set the image size that matches the input size of the model
img_size = (224, 224)

# Load the pre-trained model
model = load_model('skin_cancer_detection_model.h5')


# Function to preprocess the uploaded image (resize and normalize)
def preprocess_image(img):
    img = cv2.resize(img, img_size)  # Resize to (224, 224)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize image values to [0, 1]
    return img


# Define the app interface using Streamlit
def app():
    st.title("Easy Skin Health Checker")
    st.markdown("""
        Welcome to the Easy Skin Health Checker!
        Upload an image of your skin mole, and this app will give a preliminary assessment.
        Note: This is for informational purposes only and should not replace professional medical advice.
    """)

    # Navigation Tabs
    tabs = st.tabs(["Upload Image", "Detection Results", "Disclaimer"])

    # Image Upload Section
    with tabs[0]:
        st.header("Step 1: Upload an Image of Your Mole")
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

    # Detection Results Section
    with tabs[1]:
        st.header("Step 2: View Detection Results")

        if uploaded_file is not None:
            # Display a summary of the results in a more understandable manner
            if pred_label == 'Cancer':
                st.error(
                    f"Warning: The model has detected a high probability ({pred_prob * 100:.2f}%) that this mole could be cancerous.")
                st.markdown("""
                       ### What does this mean?
                       Our model's analysis indicates that the uploaded image shows features commonly associated with skin cancer.
                       We strongly advise you to consult a healthcare professional for further examination and diagnosis.
                   """)
            else:
                st.success(
                    f"Good News: The model has detected a low probability ({(1 - pred_prob) * 100:.2f}%) of skin cancer.")
                st.markdown("""
                       ### What does this mean?
                       Our model suggests that this mole is not likely to be cancerous based on the image provided. However, we recommend
                       regular monitoring and visiting a dermatologist if you notice any changes.
                   """)

            # Additional visual aids for clarity
            st.info("""
                   **Note**: This result is based on a machine learning model and should not replace professional medical advice. Always seek the
                   opinion of a medical expert for concerns about your health.
               """)

    # Disclaimer Section
    with tabs[2]:
        st.header("Disclaimer")
        st.markdown("""
            This app is for preliminary guidance only and should not be used as a substitute for a professional medical diagnosis.
            Please consult a healthcare provider for any concerns regarding your skin health.
        """)
        agree = st.checkbox("I understand and accept the disclaimer.")
        if not agree:
            st.warning("You must accept the disclaimer before using the app.")


# Running the app
if __name__ == '__main__':
    app()
