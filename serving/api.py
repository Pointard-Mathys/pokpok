from fastapi import FastAPI , File , UploadFile , Request , Form
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import mlflow
import mlflow
import mlflow.pyfunc
import mlflow.keras
from tensorflow.keras.models import load_model
import tensorflow as tf
import json
import pandas as pd 
import math

app = FastAPI()

ARTIFACTS_DIR = "../artifacts/"
NEW_IMAGES_DIR = os.path.join(ARTIFACTS_DIR, "new_images/") 
LABELS_FILE = os.path.join(ARTIFACTS_DIR, "labels.json")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "cnn_pokemon_model.h5")
MODEL_BACKUP_PATH = os.path.join(ARTIFACTS_DIR, "cnn_pokemon_model_backup.h5")

with open("../artifacts/pokemon_dict.json", "r", encoding="utf-8") as f:
    pokemon_dict = json.load(f)
    
df = pd.read_csv('../artifacts/pokemon(1).csv')

try:
    model = tf.keras.models.load_model("../artifacts/cnn_pokemon_model.h5")
    model_loaded = True
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    model = None
    model_loaded = False
    
def predict_image(filepath, model):
    if not os.path.exists(filepath):
        print(f"Erreur : L'image {filepath} n'existe pas.")
        return

    if model is None:
        print("Erreur : Aucun modèle chargé.")
        return
    
    img = image.load_img(filepath, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    img_to_show = mpimg.imread(filepath)
    plt.imshow(img_to_show)
    plt.axis('off')
    plt.show()

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    
    predicted_label = list(pokemon_dict.keys())[predicted_class_index]
    confidence = prediction[0][predicted_class_index]
    
    print(f"Prédiction : {predicted_label} avec une probabilité de {confidence:.2f}")
    return predicted_label , confidence

@app.get("/")
def hello_world():
    return {
        "message": "Hello World - FastAPI",
        "model_loaded": model_loaded
    }
    
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    temp_filepath = f"temp_{file.filename}"
    with open(temp_filepath, "wb") as buffer:
        buffer.write(await file.read())
    pokemon, confidence = predict_image(temp_filepath , model)
    
    if not pokemon:
        os.remove(temp_filepath)
        return {"error": "Prédiction invalide. Aucun Pokémon détecté."}
    
    if math.isnan(confidence) or math.isinf(confidence):
        confidence = 0.0  
        
    data = df[df['Nom'] == pokemon]
    if not data.empty : 
        id = data.ID.values[0]
        type1 = data.Type1.values[0]
        type2 = data.Type2.values[0] if pd.notna(data.Type2.values[0]) else None
        attack = int(data.Attaque.values[0])
        attack_spe = int(data.Attaque_spe.values[0])
        defense = int(data.Defense.values[0])
        def_spe = int(data.Defense_spe.values[0])
        speed = int(data.Vitesse.values[0])
        
        os.remove(temp_filepath)
        

        return {
        "pokemon": pokemon,
        "confidence": float(confidence),
        "id": id,
        "type1": type1,
        "type2": type2,
        "attack": attack,
        "attack_spe": attack_spe,
        "defense": defense,
        "def_spe": def_spe,
        "speed": speed
    }
    else : 
        os.remove(temp_filepath)
        return {"error" : "Pokemon not found"}
    

@app.post("/feedback/")
async def feedback(file: UploadFile = File(...), true_pokemon: str = Form(...)):
    print(f"🚀 Feedback reçu pour le Pokémon : {true_pokemon}")
    if not true_pokemon:
        return {"error": "Aucun nom de Pokémon fourni."}
    
    # Création du dossier si besoin
    os.makedirs(NEW_IMAGES_DIR, exist_ok=True)
    
    # Construire le nom de fichier et gérer les doublons
    base_filename = f"{true_pokemon}.jpg"
    filename = base_filename
    counter = 1
    while os.path.exists(os.path.join(NEW_IMAGES_DIR, filename)):
        filename = f"{true_pokemon}_{counter}.jpg"
        counter += 1
    filepath = os.path.join(NEW_IMAGES_DIR, filename)
    
    # Enregistrer l'image reçue
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())
    
    # Charger ou initialiser la liste des corrections
    corrections = []
    if os.path.exists(LABELS_FILE):
        try:
            with open(LABELS_FILE, "r", encoding="utf-8") as f:
                corrections = json.load(f)
        except json.decoder.JSONDecodeError:
            corrections = []
    
    corrections.append({"file": filename, "true_pokemon": true_pokemon})
    
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(corrections, f, indent=4)
    
    print(f"✅ Image enregistrée sous {filepath} et feedback stocké.")
    return {"message": "Feedback enregistré", "filename": filename, "true_pokemon": true_pokemon}



@app.post("/retrain/")
async def retrain_model():
    """Réentraîne le modèle avec les corrections fournies par les utilisateurs."""
    
    if not os.path.exists(LABELS_FILE):
        return {"message": "Aucune correction trouvée. Pas de réentraînement nécessaire."}
    
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        corrections = json.load(f)
    
    # On ne lance le réentraînement que si on a au moins 5 corrections
    if len(corrections) < 5:
        return {"message": f"Pas assez d'images pour le réentraînement. Actuellement {len(corrections)} corrections."}
    
    model = tf.keras.models.load_model(MODEL_PATH)
    X_train, y_train = [], []
    
    for entry in corrections:
        img_path = os.path.join(NEW_IMAGES_DIR, entry["file"])
        true_label = entry["true_pokemon"]
        if os.path.exists(img_path):
            label = pokemon_dict.get(true_label)
            if label is None:
                print(f"Label inconnu pour le Pokémon : {true_label}. Correction ignorée.")
                continue
            try:
                img = image.load_img(img_path, target_size=(128, 128))
            except Exception as e:
                print(f"Erreur lors de la lecture de l'image {img_path}: {e}")
                continue
            img_array = image.img_to_array(img) / 255.0
            X_train.append(img_array)
            y_train.append(label)
    
    if len(X_train) == 0:
        return {"message": "Aucune image valide trouvée pour le réentraînement."}
    
    X_train = np.array(X_train)
    # Déduire le nombre de classes à partir du dictionnaire
    num_classes = max(pokemon_dict.values()) + 1
    y_train = to_categorical(y_train, num_classes=num_classes)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    model.save(MODEL_BACKUP_PATH)  # Sauvegarder le modèle existant
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)
    loss, accuracy = model.evaluate(X_val, y_val)
    
    # Comparaison avec l'ancien modèle
    old_model = tf.keras.models.load_model(MODEL_BACKUP_PATH)
    _, old_accuracy = old_model.evaluate(X_val, y_val)
    
    if accuracy > old_accuracy:
        model.save(MODEL_PATH)
        message = "✅ Nouveau modèle sauvegardé !"
    else:
        message = "❌ Le modèle réentraîné est moins bon, annulation..."
        os.remove(MODEL_PATH)
        os.rename(MODEL_BACKUP_PATH, MODEL_PATH)
    
    os.remove(LABELS_FILE)  # Suppression des corrections utilisées
    
    return {"message": message, "new_accuracy": accuracy, "old_accuracy": old_accuracy}
