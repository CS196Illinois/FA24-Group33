import streamlit as st 
# Title of the app 
st.title("Basic Streamlit App") 
# Slider to select a number 
number = st.slider("Choose a number", 0, 100) 
# Text input 
text = st.text_input("Enter your name") # Display the chosen number and name st.write(f"Hello, {text}! You selected the number {number}.") 
# Simple calculation based on the number 
st.write(f"Twice the selected number is {number * 2}.")
