import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

df = pd.read_csv("dataset/HAM10000_metadata.csv")

image_dir1 = "dataset/HAM10000_images_part_1"
image_dir2 = "dataset/HAM10000_images_part_2"

image_paths = {}

for folder in [image_dir1, image_dir2]:
    for file in os.listdir(folder):
        image_paths[file.split(".")[0]] = os.path.join(folder, file)

df["path"] = df["image_id"].map(image_paths)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["dx"]
)

print("Train:", len(train_df))
print("Test:", len(test_df))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="path",
    y_col="dx",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

validation_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="path",
    y_col="dx",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(128, activation="relu")(x)

x = Dropout(0.5)(x)

predictions = Dense(
    7,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()
# history = model.fit(
#     train_generator,
#     validation_data=validation_generator,
#     epochs=2
# )

import os

os.makedirs("models", exist_ok=True)

model.save("models/skin_cancer_model.h5")

print("Model Saved Successfully!")