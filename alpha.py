import streamlit as st
import os
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
import cv2
import gtts
import numpy as np
from keras.models import load_model
from gtts import gTTS
from io import BytesIO
import base64

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        font-size: 40px;
        font-family: 'Times New Roman', Times, serif;
        color: #FFFFFF; 
    }
    .centered-text {
        text-align: center;
        font-size: 30px;
        font-family: 'Times New Roman', Times, serif;
        color: #000000; 
    }
    .normal-text {
        font-size: 24px;
        font-family: 'Times New Roman', Times, serif;
        color: #000000; 
    }
    body {
        background-color: rgba(255, 255, 255);
        background-size: cover;
        background-attachment: fixed;
        font-family: 'Times New Roman', Times, serif;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.8); 
        padding: 20px;
        border-radius: 10px;
    }
    h1, h2, h3, h4, p {
        color: black !important; 
    }
    .stButton>button {
        background-color: #008CBA; 
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #005f6b; 
    }
    .centered-checkbox-label {
        font-family: 'Times New Roman', Times, serif;
        font-size: 24px;
        color: #000000; 
    }
    .custom-warning {
            font-family: 'Times New Roman', Times, serif;
            font-size: 24px;
            color: #FF0000; 
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Set the image size that matches the input size of the model
img_size = (224, 224)

# Load the pre-trained model
model = load_model('skin_cancer_detection_model.h5')


# Function to preprocess the uploaded image (resize and normalize)
def preprocess_image(img):
    img = cv2.resize(img, img_size)  # Resize to (224, 224)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  
    return img


def app():
    tabs = st.tabs(["Image uploader", "GitHub Repository Credits", "Contact us"])

    with tabs[0]:
        st.markdown('<p class="centered-title">Skin Cancer Detection</p>', unsafe_allow_html=True)
        st.markdown("""
        <p class="centered-text">
            Please upload a picture of your mole to attain an analysis. Note that you should accept the medical disclaimer prior to submitting images.
        </p>
        <p class="centered-text">
            Please note that this app does not provide any medical advice that can substitute a medical practioner. In cases of suspicion of melanoma, please consult a specialist. 
            The goal of this website is to ensure that basic, preliminary screening resources are publicly available and accessible. 
        </p>
        
        """, unsafe_allow_html=True)
        introduction = "Welcome to the Easy Skin Health Checker! Upload an image of your skin mole, and this app will give a preliminary assessment. Please upload a picture of your mole to attain an analysis. Note that you should accept the medical disclaimer prior to submitting images. The medical disclaimer is as follows please note that this app does not provide any medical advice that can substitute a medical practioner. In cases of suspicion of melanoma, please consult a specialist. The goal of this website is to ensure that basic, preliminary screening resources are publicly available and accessible. You must accept the disclaimer before using the app. Your images will not be processed until this box is checked. You can uncheck this box to withdraw consent."
        
        if st.button('Read Aloud'):
            tts = gTTS(text=introduction, lang='en', slow=False)
            tts.save("output.mp3")
            st.audio("output.mp3")

        agree = st.checkbox("")

        st.markdown('<p class="centered-checkbox-label">I understand and accept the disclaimer.</p>', unsafe_allow_html=True)

        if not agree:
            st.markdown('<p class="custom-warning">You must accept the disclaimer before using the app.</p>', unsafe_allow_html=True)
            st.markdown('<p class="custom-warning">Your images will not be processed until this box is checked. You can uncheck this box to withdraw consent.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="custom-warning">You have accepted the disclaimer. You can uncheck this box to withdraw consent.</p>', unsafe_allow_html=True) 



        st.markdown('<p class="centered-title">Upload an image of your mole</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader('.', type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None and agree:
            img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
            st.image(img, caption='Imag e uploaded by user', use_column_width=True)
            img = preprocess_image(img)
            pred = model.predict(img)
            pred_prob = pred[0][0]
            if pred_prob >= 0.5:
                st.markdown('<p class="custom-warning"> According to the model, there are signs the uploaded image is cancerous.</p>', unsafe_allow_html=True)
            else: 
                st.markdown('<p class="custom-warning"> According to the model, there are signs the uploaded image is NOT cancerous. In cases of worry, please consult a medical provider: this is NOT a substitute for a professional.</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="custom-warning"> Probability Of Skin Cancer: {pred_prob:.2f} </p>', unsafe_allow_html=True)
            st.markdown('<p class="centered-title">Explanation of Detection Results</p>', unsafe_allow_html=True)
        
        if uploaded_file is not None and agree:
            if pred_prob >= 0.5:
                st.markdown("""
                <p class="normal-text">
                Please note that this application does not guarantee an accurate diagnosis. It is recommended that you make an appointment with your local dermatologist to discuss the steps forward. This result is based on a machine learning model and should not replace professional medical advice. Always seek the opinion of a medical expert for concerns about your health.
                </p>
                """, unsafe_allow_html=True)
                result = "According to the model, there are signs the uploaded image is cancerous. In cases of worry, please consult a medical provider: this is NOT a substitute for a professional. Please note that this application does not guarantee an accurate diagnosis. It is recommended that you make an appointment with your local dermatologist to discuss the steps forward. This result is based on a machine learning model and should not replace professional medical advice. Always seek the opinion of a medical expert for concerns about your health."
        
                if st.button('Read Aloud Result'):
                    tts = gTTS(text= result, lang='en', slow=False)
                    tts.save("output.mp3")
                    st.audio("output.mp3")
            else:
                st.markdown("""
                <p class="normal-text">
                Please note that this application does not guarantee an accurate diagnosis. It is still recommended that you make an appointment with your local dermatologist if you have cause for suspicion. This result is based on a machine learning model and should not replace professional medical advice. Always seek the opinion of a medical expert for concerns about your health.
                </p>
                   """, unsafe_allow_html=True)
                result = "According to the model, there are signs the uploaded image is NOT cancerous. In cases of worry, please consult a medical provider: this is NOT a substitute for a professional. Please note that this application does not guarantee an accurate diagnosis. It is recommended that you make an appointment with your local dermatologist to discuss the steps forward. This result is based on a machine learning model and should not replace professional medical advice. Always seek the opinion of a medical expert for concerns about your health."
        
                if st.button('Read Aloud Result'):
                    tts = gTTS(text= result, lang='en', slow=False)
                    tts.save("output.mp3")
                    st.audio("output.mp3")


            
    with tabs[1]:

        st.markdown("""
                <p class="centered-title">
                    GitHub Repository Credits: Skin Cancer Detection Model by Deepankar Varma
                </p>             
        """, unsafe_allow_html=True)
        content = "GitHub Repository Credits: Skin Cancer Detection Model by Deepankar Varma. This project uses resources from the following GitHub repository: Skin Cancer Detection Model Repository. This repository provides a machine learning model trained for the detection of Skin Cancer. Leveraging OpenCV for image processing and Tensorflow with Keras for building the model, this respository provides a key tool to aid in preliminary detection of melanoma. Our project recognizes the potential impact of this repository and aims to integrate an accessible, user-friendly interfact for the use of this model. To run the code in this repository, you'll need the following dependencies: Python 3.x, TensorFlow, Keras, NumPy, OpenCV"
        if st.button('Read Aloud Credits'):
            tts = gTTS(text=content, lang='en', slow=False)
            tts.save("output.mp3")
            st.audio("output.mp3")
        st.markdown("""
                <p class="normal-text">
                    This project uses resources from the following GitHub repository:
                    <a href="https://github.com/deepankarvarma/Skin-Cancer-Detection--OpenCV-TensorFlow-Keras.git" target="_blank">
                    Skin Cancer Detection Model Repository
                </a>
                </p>
                <p class="normal-text">
                    This repository provides a machine learning model trained for the detection of Skin Cancer. Leveraging OpenCV for image processing and Tensorflow with Keras for building the model, this respository provides a key tool to aid in preliminary detection of melanoma. Our project recognizes the potential impact of this repository and aims to integrate an accessible, user-friendly interfact for the use of this model.  

                </p>
                <p class="normal-text">
                    To run the code in this repository, you'll need the following dependencies: Python 3.x, TensorFlow, Keras, NumPy, OpenCV

                </p>   
                          
                """, unsafe_allow_html=True)



    with tabs[2]:
        st.markdown('<p class="centered-title">LinkedIn Contact Information</p>', unsafe_allow_html=True)

        profiles = [
            ("LinkedIn_images/Pras.jpg" , "Prasheetha (Pras) Bairwa", "https://www.linkedin.com/in/prasheetha-bairwa/", "Computer Science @ UIUC"),
            ("LinkedIn_images/Will.jpg" , "William Hubbe", "https://www.linkedin.com/in/william-hubbe-927061338/", "Computer Science @ UIUC"),
            ("LinkedIn_images/James.jpg", "Zhiheng (James) Weng", "https://www.linkedin.com/in/james-weng-545721337/", "Division of General Studies @ UIUC"),
            ("LinkedIn_images/Abhi.jpeg", "Abhiram (Abhi) Chiduruppa", "https://www.linkedin.com/in/abhiram-chiduruppa-b5606b283", "Computer Science @ UIUC"),
            ("LinkedIn_images/Zack.jpg", "Yantao Lin", "https://www.linkedin.com/in/言涛-林-a91299338/", "Computer Science + Maths @ UIUC"),
            ("LinkedIn_images/Aasiya.jpg", "Aasiya Memon", "https://www.linkedin.com/in/aasiya-memon/", "Computer Science + Music @ UIUC")
        ]

        rows = [st.columns(3) for _ in range(2)] 
        for i, (img_url, name, linkedin_url, description) in enumerate(profiles):
            row = rows[i // 3]
            with row[i % 3]: 
                st.image(img_url, use_column_width=True)
                st.markdown(
                    f"""
                    <div class="centered-text">
                        <a href="{linkedin_url}" style="color: inherit; text-decoration: none;"><b>{name}</b></a>
                        <p>{description}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )                

if __name__ == '__main__':
    app()