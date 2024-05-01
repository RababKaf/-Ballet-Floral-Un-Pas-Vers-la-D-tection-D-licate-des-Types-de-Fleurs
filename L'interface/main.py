import streamlit as st
import pickle
from PIL import Image
from ML_Fen import *
import pandas as pd


#chargement de model :
with open('MyModel/modelCNNCV.p', 'rb') as file:
    model = pickle.load(file)
set_background("bg.png")
st.title(" ")
st.header(" ")

#Chargement d'image :
file=st.file_uploader(' ', type=['jpeg', 'jpg', 'png'])

data = {
    'Fleur': ['Amirillis', 'Forget me nots', 'Jasmin', 'Lotus', 'Osteospermum', 'Pansy', 'Daisy', 'Dendelions', 'Rose', 'Sunflowers', 'Tulips'],
    'Période de floraison': ['Hiver-Printemps', 'Printemps', 'Printemps-Été', 'Été', 'Printemps-Été', 'Printemps-Été', 'Printemps-Été', 'Printemps-Été', 'Printemps-Été', 'Été', 'Printemps'],
    'Origine': ['Amérique du Sud', 'Europe', 'Asie', 'Asie', 'Afrique du Sud', 'Europe', 'Europe', 'Europe', 'Asie', 'Amérique du Nord', 'Europe'],
    'Taille': ['Grande', 'Petite', 'Moyenne', 'Grande', 'Petite', 'Moyenne', 'Petite', 'Petite', 'Moyenne', 'Grande', 'Moyenne'],
    'Exposition au soleil': ['Plein soleil', 'Mi-ombre', 'Plein soleil', 'Plein soleil', 'Plein soleil', 'Plein soleil', 'Plein soleil', 'Plein soleil', 'Plein soleil', 'Plein soleil', 'Plein soleil'],
    'Type de sol': ['Bien drainé', 'Humide', 'Bien drainé', 'Aquatique', 'Bien drainé', 'Bien drainé', 'Bien drainé', 'Bien drainé', 'Bien drainé', 'Bien drainé', 'Bien drainé'],
    'Comment fleurir': [
        'Plantez les bulbes dans un sol bien drainé à l\'automne. Arrosez régulièrement et placez dans un endroit ensoleillé.',
        'Semez les graines directement dans le sol au printemps. Gardez le sol humide jusqu\'à ce que les plantes soient établies.',
        'Plantez dans un sol bien drainé et ensoleillé. Arrosez régulièrement pendant la croissance active.',
        'Plantez dans un bassin d\'eau peu profond en plein soleil. Fertilisez régulièrement pendant la saison de croissance.',
        'Plantez dans un sol bien drainé en plein soleil. Arrosez régulièrement pendant la croissance active.',
        'Plantez dans un sol riche en matière organique en plein soleil à mi-ombre. Arrosez régulièrement.',
        'Plantez dans un sol bien drainé en plein soleil. Arrosez régulièrement pendant la croissance active.',
        'Les pissenlits poussent dans presque tous les sols et conditions. Arrosez régulièrement.',
        'Plantez dans un sol bien drainé en plein soleil. Arrosez régulièrement et fertilisez au printemps.',
        'Plantez les graines directement dans le sol en plein soleil. Arrosez régulièrement pendant la croissance active.',
        'Plantez les bulbes à l\'automne dans un sol bien drainé en plein soleil à mi-ombre. Arrosez régulièrement.'
    ],
    'Comment refleurir': [
        'Après la floraison, coupez les fleurs fanées et continuez à arroser régulièrement. Laissez les feuilles se faner naturellement.',
        'Après la floraison, coupez les fleurs fanées pour encourager une nouvelle floraison. Arrosez régulièrement.',
        'Taillez légèrement après la floraison pour favoriser de nouvelles pousses. Continuez à arroser régulièrement.',
        'Éliminez les feuilles et les tiges mortes après la floraison. Continuez à fertiliser et à contrôler les mauvaises herbes.',
        'Taillez légèrement après la floraison pour encourager une nouvelle croissance. Fertilisez modérément.',
        'Retirez les fleurs fanées pour encourager une nouvelle floraison. Fertilisez toutes les quelques semaines.',
        'Taillez après la floraison pour favoriser de nouvelles pousses. Fertilisez légèrement au printemps.',
        'Les pissenlits sont des plantes vivaces et peuvent repousser d\'eux-mêmes chaque année.',
        'Taillez après la floraison pour favoriser de nouvelles pousses. Fertilisez modérément.',
        'Coupez les têtes de fleurs fanées pour encourager de nouvelles fleurs. Fertilisez modérément.',
        'Après la floraison, coupez les fleurs fanées pour encourager de nouvelles fleurs. Fertilisez au printemps.'
    ] ,

'Nom en arabe': ['الأماريليس', 'لا تنسني', 'الياسمين', 'اللوتس', 'أوستيوسبرموم', 'القرنفل الوجه', 'البابونج', 'الهندباء', 'وردة', 'عباد الشمس','الزنبق'],
}

df = pd.DataFrame(data)

if file is not None:
    # Charger l'image
    image = Image.open(file).convert('RGB')
    # Diviser la mise en page en colonnes
    col1, col2 = st.columns(2)
    # Afficher l'image dans la première colonne avec une taille spécifiée
    with col1:
        st.image(image, caption='Image téléchargée', use_column_width=True, width=800)

        # Prédiction
        Prediction = Classification(image, model)
        st.write("##  La prédiction :", Prediction[0])

    flower_info = df[df['Fleur'] == Prediction[0]]

    # Vérifier si la fleur a été trouvée dans le DataFrame
    if not flower_info.empty:
        # Récupérer les informations sur la fleur
        period_flowering = flower_info['Période de floraison'].values[0]
        Exposition_au_soleil = flower_info['Exposition au soleil'].values[0]
        soil_type = flower_info['Type de sol'].values[0]
        Taille = flower_info["Taille"].values[0]
        Origine = flower_info["Origine"].values[0]
        refleurir = flower_info["Comment refleurir"].values[0]
        fleurir = flower_info["Comment fleurir"].values[0]
        NomArabe = flower_info["Nom en arabe"].values[0]

        # Afficher les informations sur la fleur dans la deuxième colonne
        with col2:
            # Prédiction
            Prediction = Classification(image, model)
            st.write(" La prédiction :", Prediction[0])
            st.write("Informations sur la fleur prédite :")
            st.write(f"- Période de floraison : {period_flowering}")
            st.write(f"- Type de sol : {soil_type}")
            st.write(f"- Exposition au soleil : {Exposition_au_soleil}")
            st.write(f"- Taille : {Taille}")
            st.write(f"- Origine : {Origine}")
            st.write(f"- Comment fleurir : {fleurir}")
            st.write(f"- Comment refleurir: {refleurir}")
            st.write(f"- Nom en arabe: {NomArabe}")
    else:
        st.write("Fleur non trouvée dans la base de données.")





