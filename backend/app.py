from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
CORS(app)

model = load_model("models/skin_cancer_model.h5")

class_names = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/predict", methods=["POST"])
def predict():

    print("Prediction request received")

    file = request.files["image"]

    upload_path = "temp.jpg"
    file.save(upload_path)

    print("Image saved")

    img = image.load_img(upload_path, target_size=(224,224))

    print("Image loaded")

    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    print("Prediction completed")

    predicted_code = class_names[np.argmax(prediction)]
    predicted_class = class_mapping[predicted_code]
    confidence = float(np.max(prediction) * 100)

    return jsonify({
        "class": predicted_class,
        "confidence": round(confidence,2)
    })
    
    
class_mapping = {
    "nv": "Melanocytic Nevus (Benign)",
    "mel": "Melanoma (Malignant)",
    "bcc": "Basal Cell Carcinoma (Malignant)",
    "akiec": "Actinic Keratosis (Malignant)",
    "bkl": "Benign Keratosis (Benign)",
    "df": "Dermatofibroma (Benign)",
    "vasc": "Vascular Lesion (Benign)"
}
if __name__ == "__main__":
    app.run(debug=True)
    
