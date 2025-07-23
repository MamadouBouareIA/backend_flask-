from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

try:
    model = tf.keras.models.load_model('damage_comparison_model.h5')
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    # Gérer l'erreur, par exemple en arrêtant l'application

# Vues autorisées
ALLOWED_VIEWS = ['front', 'back', 'left', 'right']

#  Vérifie si le nom de l’image contient une vue autorisée
def is_valid_view(filename):
    return any(view in filename.lower() for view in ALLOWED_VIEWS)

#  Prétraitement d'image
def preprocess_image(file):
    image = Image.open(file).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)  # (1, 224, 224, 3)

#  Endpoint principal
@app.route('/predict', methods=['POST'])
def compare_images():
    if 'image_T0' not in request.files or 'image_T1' not in request.files:
        return jsonify({'error': "Les deux fichiers 'image_T0' et 'image_T1' sont requis."}), 400

    file_T0 = request.files['image_T0']
    file_T1 = request.files['image_T1']

    #Vérification du type de vue
    if not is_valid_view(file_T0.filename) or not is_valid_view(file_T1.filename):
        return jsonify({
            'error': "Les fichiers doivent correspondre à des vues autorisées : front, back, left, right."
        }), 403

    #  Prétraitement
    image_T0 = preprocess_image(file_T0)
    image_T1 = preprocess_image(file_T1)

    #  Prédiction
    prediction = model.predict([image_T0, image_T1])
    print("Prediction:", prediction)

    # Gérer les différentes formes de prédiction
    if isinstance(prediction, dict):
        prob = list(prediction.values())[0][0][0] if isinstance(list(prediction.values())[0], np.ndarray) else list(prediction.values())[0]
    elif isinstance(prediction, np.ndarray):
        prob = prediction[0][0] if len(prediction.shape) > 1 else prediction[0]
    else:
        return jsonify({'error': "Format de prédiction inconnu"}), 500

    anomaly = prob > 0.5

    return jsonify({
        'anomaly_detected': bool(anomaly),
        'damage_probability': round(float(prob), 4),
        'view_T0': file_T0.filename,
        'view_T1': file_T1.filename
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
