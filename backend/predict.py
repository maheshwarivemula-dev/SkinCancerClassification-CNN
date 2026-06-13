import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

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

img_path = "test.jpg"   # change later

img = image.load_img(img_path, target_size=(224,224))
img = image.img_to_array(img)
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print("Class:", predicted_class)
print("Confidence:", confidence)