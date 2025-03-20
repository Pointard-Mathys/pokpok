import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import json

# URL de l'API FastAPI pour la prédiction
API_URL = "http://fastapi_app:8080/predict/"  # Assure-toi que FastAPI tourne bien
API_URL_FEEDBACK = "http://fastapi_app:8080/feedback/"
API_URL_RETRAIN = "http://fastapi_app:8080/retrain/"

if "feedback_count" not in st.session_state:
    st.session_state["feedback_count"] = 0
if "feedback_requested" not in st.session_state:
    st.session_state["feedback_requested"] = False


# Fonction pour envoyer l'image à FastAPI et récupérer la prédiction
def get_pokemon_prediction(image_file):
    files = {"file": image_file}
    response = requests.post(API_URL, files=files)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Erreur de prédiction"}

def send_feedback(image_file, true_pokemon):
    files = {"file": image_file}
    data = {"true_pokemon": true_pokemon}
    response = requests.post(API_URL_FEEDBACK, files=files, data=data)
    
    print(f"🔍 Réponse API Feedback : {response.status_code} - {response.text}")  # Ajout du log

    return response.status_code == 200


# 📌 Fonction pour déclencher le réentraînement du modèle
def retrain_model():
    response = requests.post(API_URL_RETRAIN)
    if response.status_code == 200:
        return response.json()
    return {"error": "Impossible de réentraîner le modèle"}

# Fonction pour afficher le radar chart
def plot_radar_chart(stats):
    labels = ["Attaque", "Attaque Spé", "Défense", "Défense Spé", "Vitesse"]
    values = [
        stats["attack"], stats["attack_spe"], stats["defense"],
        stats["def_spe"], stats["speed"]
    ]

    # Ajouter la première valeur à la fin pour fermer le pentagone
    values.append(values[0])
    
    # Générer les angles pour un pentagone
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles.append(angles[0])  # Fermer le pentagone

    # Création du radar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)  # Décalage pour aligner correctement
    ax.set_theta_direction(-1)  # Inversion pour correspondre aux conventions des radars

    # Dessiner le polygone
    ax.fill(angles, values, color="red", alpha=0.4)
    ax.plot(angles, values, color="red", linewidth=2)

    # Ajouter les labels des axes
    ax.set_xticks(angles[:-1])  # Mettre uniquement les angles des 5 valeurs
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold", ha="center", va="center")

    # Ajustement pour que le titre ne soit pas superposé aux labels
    ax.set_title("Statistiques du Pokémon", fontsize=14, fontweight="bold", pad=30)

    # Enlever les valeurs des ticks radiaux pour éviter la confusion
    ax.set_yticklabels([])

    # Afficher le graphique dans Streamlit
    st.pyplot(fig)


# Interface Streamlit
st.title("Pokédex 🔥")

uploaded_file = st.file_uploader("Upload une image de Pokémon", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Image du Pokémon uploadée", use_column_width=True)
    with st.spinner("Analyse de l'image..."):
        prediction = get_pokemon_prediction(uploaded_file)
    
    if not prediction.get("error"):
        pokemon_name = prediction["pokemon"]
        confidence = prediction["confidence"]
        st.success(f"Pokémon détecté : **{pokemon_name}** avec une confiance de **{confidence*100:.2f}%**")
        
        st.subheader("Statistiques :")
        st.write(f"**ID :** {prediction['id']}")
        st.write(f"**Type 1 :** {prediction['type1']}")
        if prediction["type2"] is not None:
            st.write(f"**Type 2 :** {prediction['type2']}")
        st.write(f"**Attaque :** {prediction['attack']}")
        st.write(f"**Attaque Spé :** {prediction['attack_spe']}")
        st.write(f"**Défense :** {prediction['defense']}")
        st.write(f"**Défense Spé :** {prediction['def_spe']}")
        st.write(f"**Vitesse :** {prediction['speed']}")
        
        stats = {"attack": prediction["attack"],
                 "attack_spe": prediction["attack_spe"],
                 "defense": prediction["defense"],
                 "def_spe": prediction["def_spe"],
                 "speed": prediction["speed"]}
        plot_radar_chart(stats)
        
        st.subheader("La prédiction est-elle correcte ?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Oui, c'est correct"):
                st.success("Merci pour votre validation ! 😊")
        with col2:
            if st.button("❌ Non, c'est incorrect"):
                st.session_state.feedback_requested = True
        
        if st.session_state.get("feedback_requested", False):
            true_pokemon = st.text_input("Entrez le vrai nom du Pokémon :")
            if st.button("Envoyer la correction"):
                if true_pokemon:
                    uploaded_file.seek(0)  # Réinitialiser le curseur de l'image
                    with st.spinner("Envoi du feedback..."):
                        feedback_success = send_feedback(uploaded_file, true_pokemon)
                    if feedback_success:
                        st.success("✅ Feedback enregistré avec succès !")
                        st.session_state.feedback_requested = False
                        st.session_state.feedback_count += 1
                        st.write("Nombre de corrections soumises : ", st.session_state.feedback_count)
                        # Si on a atteint 5 feedbacks, déclencher le réentraînement
                        if st.session_state.feedback_count >= 10:
                            with st.spinner("Réentraînement du modèle en cours..."):
                                retrain_response = retrain_model()
                            if "error" not in retrain_response:
                                st.success("🎉 " + retrain_response.get("message", "Réentraînement terminé !"))
                                # Réinitialiser le compteur
                                st.session_state.feedback_count = 0
                            else:
                                st.error("❌ Erreur lors du réentraînement du modèle.")
                        else:
                            st.info(f"Pas assez d'images pour le réentraînement. ({st.session_state.feedback_count} sur 5)")
                    else:
                        st.error("❌ Erreur lors de l'enregistrement du feedback.")
                else:
                    st.warning("Veuillez entrer le vrai nom du Pokémon avant d'envoyer.")
    else:
        st.error("Erreur dans la prédiction. Vérifie que FastAPI tourne bien !")