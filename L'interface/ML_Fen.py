import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import base64


#Extracton des caracteristqiue pour une seule image :

def extract_features_single_image(image):
    # Charger le modèle ResNet50 pré-entraîné sans les couches fully-connected
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    features = []
    # Charger et prétraiter l'image
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    # Extraire les caractéristiques de l'image en utilisant ResNet50
    features.append(model.predict(img_array).flatten())
    features = np.array(features)
    return features
def Classification(image , model  ):
    features = extract_features_single_image(image)
    prediction = model.predict(features)
    #print(prediction)

    return prediction

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
             color: rgb(16, 81, 78);
             font-weight: bold;
             font-size=30px;
             
             
        }}
        .st-ct {{
            color: white;  /* Changer la couleur du texte global */
        }}
        .st-eb {{
            color: blue;   /* Changer la couleur du titre */
        }}
        </style>
    """

    st.markdown(style, unsafe_allow_html=True)
    st.markdown("<h1 style='color: #D64057;'>Floral Ballet: A Step Towards Delicate Flower Detection</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #D64057; text-align: center;'>Upload an image</h3>",unsafe_allow_html=True)



