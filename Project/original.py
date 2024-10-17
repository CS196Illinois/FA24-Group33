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


def app():
    st.title("Skin Cancer Detection")
    st.markdown("""
        Welcome to the Easy Skin Health Checker!
        Upload an image of your skin mole, and this app will give a preliminary assessment.
        Note: This is for informational purposes only and should not replace professional medical advice.
    """)

    tabs = st.tabs(["Image uploader", "Detection Result",  "GitHub Repository Credits"])

    with tabs[0]:
        st.markdown("""
            Please note that this app does not provide any medical advice that can substitute a medical practioner. In cases of suspicion of melanoma, please consult a specialist. 
            The goal of this website is to ensure that basic, preliminary screening resources are publicly available and accessible. 

        """)
        agree = st.checkbox("I understand and accept the disclaimer.")
        if not agree:
            st.warning("You must accept the disclaimer before using the app.")
            st.warning(
                "Your images will not be processed until this box is checked. You can uncheck this box to withdraw consent.")

        st.header("Upload an Image of Your Mole")
        uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None and agree:
            img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
            st.image(img, caption='Image uploaded by user', use_column_width=True)
            img = preprocess_image(img)
            pred = model.predict(img)

    with tabs[1]:
        st.header("View Detection Results")

        if uploaded_file is not None:
            # Display a summary of the results in a more understandable manner
            pred_label = 'According to the model, there are signs the uploaded image is cancerous. Please consult a medical provider.' if \
                pred[0][
                    0] > 0.5 else 'According to the model, there are signs the uploaded image is NOT cancerous. In cases of worry, please consult a medical provider: this is NOT a substitute for a professional.'
            pred_prob = pred[0][0]
            st.write(f'Prediction: {pred_label}')
            st.write(f'Probability Of Skin Cancer: {pred_prob:.2f}')
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
                    f"Good News: The model has detected a low probability ({(pred_prob) * 100:.2f}%) of skin cancer.")
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

if __name__ == '__main__':
    app()