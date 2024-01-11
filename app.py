import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load your trained model
model = load_model('mri_breast_cancer_model_DenseNet201.h5')

def preprocess_image(image, target_size):
    # Preprocessing steps as per your model's requirement
    return processed_image

st.title('Breast Cancer Classification using MRI Images')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = preprocess_image(uploaded_file, target_size=(224, 224))
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Predicting...")
    prediction = model.predict(image)
    
    # Display the prediction
    if prediction[0][0] > 0.5:
        st.write("The MRI scan is likely Malignant")
    else:
        st.write("The MRI scan is likely Benign")

