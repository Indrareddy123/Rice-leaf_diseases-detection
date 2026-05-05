# Rice-leaf_diseases-detection
import kagglehub
import shutil

# Download dataset
path = kagglehub.dataset_download("anshulm257/rice-disease-dataset")
print("Dataset downloaded to:", path)

# Copy to working directory
shutil.copytree(path, "/content/rice_dataset", dirs_exist_ok=True)

!ls /content/rice_dataset


import os

base = "/content/rice_dataset/Rice_Leaf_AUG"

rename_map = {
    "Bacterial Leaf Blight": "bacterial_leaf_blight",
    "Brown Spot": "brown_spot",
    "Healthy Rice Leaf": "healthy_rice_leaf",
    "Leaf Blast": "leaf_blast",
    "Leaf scald": "leaf_scald",
    "Sheath Blight": "sheath_blight"
}

for old, new in rename_map.items():
    os.rename(f"{base}/{old}", f"{base}/{new}")

!ls /content/rice_dataset/Rice_Leaf_AUG


import os, random, shutil
from pathlib import Path

SRC = "/content/rice_dataset/Rice_Leaf_AUG"
OUT = "/content/rice_split"
random.seed(42)

os.makedirs(OUT, exist_ok=True)

for cls in os.listdir(SRC):
    images = os.listdir(f"{SRC}/{cls}")
    random.shuffle(images)

    n = len(images)
    train = images[:int(n*0.7)]
    val   = images[int(n*0.7):int(n*0.85)]
    test  = images[int(n*0.85):]

    for folder, lst in [("train", train), ("val", val), ("test", test)]:
        dst = Path(OUT)/folder/cls
        dst.mkdir(parents=True, exist_ok=True)
        for img in lst:
            shutil.copy(f"{SRC}/{cls}/{img}", f"{dst}/{img}")

print("DONE SPLITTING!")


from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH = 32

train_gen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
).flow_from_directory(
    "/content/rice_split/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical'
)

val_gen = ImageDataGenerator(
    rescale=1/255.0
).flow_from_directory(
    "/content/rice_split/val",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode='categorical'
)

class_names = list(train_gen.class_indices.keys())
print("Classes:", class_names)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    MaxPooling2D(),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    MaxPooling2D(),

    GlobalAveragePooling2D(),

    Dense(256, activation='relu'),
    Dropout(0.4),

    Dense(len(class_names), activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()


history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20
)


from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_gen = ImageDataGenerator(rescale=1/255.0).flow_from_directory(
    "/content/rice_split/test",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)


test_loss, test_acc = model.evaluate(test_gen)
print("Final Test Accuracy:", test_acc)
print("Final Test Loss:", test_loss)


import matplotlib.pyplot as plt

# 1. Accuracy Graph
plt.figure(figsize=(8,6))
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# 2. Loss Graph
plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
