import pandas as pd

df = pd.read_csv("dataset/HAM10000_metadata.csv")

print(df.shape)
print(df.head())

import os

image_dir1 = "dataset/HAM10000_images_part_1"
image_dir2 = "dataset/HAM10000_images_part_2"

image_paths = {}

for folder in [image_dir1, image_dir2]:
    for file in os.listdir(folder):
        image_paths[file.split(".")[0]] = os.path.join(folder, file)

print("Total Images:", len(image_paths))

df["path"] = df["image_id"].map(image_paths)

print(df[["image_id", "path"]].head())


from PIL import Image
import matplotlib.pyplot as plt

sample_path = df["path"].iloc[0]

img = Image.open(sample_path)

plt.imshow(img)
plt.axis("off")
plt.show()

import numpy as np

img_array = np.array(img)

print(img_array.shape)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df["label"] = encoder.fit_transform(df["dx"])

print(df[["dx","label"]].head())

for i, cls in enumerate(encoder.classes_):
    print(i, ":", cls)
    
from tensorflow.keras.preprocessing.image import load_img, img_to_array

print("Import Successful")

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

IMG_SIZE = 224

X = []

for path in df["path"]:

    img = load_img(
        path,
        target_size=(224,224)
    )

    img = img_to_array(img)

    img = img / 255.0

    X.append(img)

X = np.array(X)

print(X.shape)

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y = encoder.fit_transform(df["dx"])

print(y[:10])

from tensorflow.keras.utils import to_categorical

y = to_categorical(y)

print(y.shape)