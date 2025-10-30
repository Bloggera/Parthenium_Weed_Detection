import os
import shutil
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import hashlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

DATASET_DIR = "dataset"
CLEANED_DIR = "cleaned_data"
IMG_SIZE = (224, 224)

def remove_duplicates_and_resize():
    print("Cleaning and resizing images...")
    seen_hashes = set()
    os.makedirs(CLEANED_DIR, exist_ok=True)

    for cls in ["parthenium", "negative"]:
        input_dir = os.path.join(DATASET_DIR, cls)
        images = os.listdir(input_dir)
        random.shuffle(images)

        # split 80-20
        split_idx = int(len(images)*0.8)
        splits = {"train": images[:split_idx], "val": images[split_idx:]}

        for split, files in splits.items():
            output_dir = os.path.join(CLEANED_DIR, split, cls)
            os.makedirs(output_dir, exist_ok=True)

            for f in tqdm(files, desc=f"{cls}-{split}"):
                try:
                    img_path = os.path.join(input_dir, f)
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize(IMG_SIZE)

                    # hash to remove dupes
                    h = hashlib.md5(np.array(img)).hexdigest()
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)

                    img.save(os.path.join(output_dir, f))
                except:
                    continue
    print("Cleaning complete.\n")

remove_duplicates_and_resize()
def train_model():
    train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20,
                                   width_shift_range=0.2, height_shift_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True)
    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        os.path.join(CLEANED_DIR, "train"), target_size=IMG_SIZE,
        batch_size=16, class_mode="binary")
    val_data = val_gen.flow_from_directory(
        os.path.join(CLEANED_DIR, "val"), target_size=IMG_SIZE,
        batch_size=16, class_mode="binary")

    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(224, 224, 3))
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    print("Training model...")
    model.fit(train_data, validation_data=val_data, epochs=10)
    model.save("parthenium_detector.h5")
    print("Model saved as parthenium_detector.h5")

train_model()
