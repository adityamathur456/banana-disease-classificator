import os
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input

# ---------------------------
# Clear session
# ---------------------------
tf.keras.backend.clear_session()

# ---------------------------
# Paths
# ---------------------------
DATA_DIR = "/kaggle/input/Banana Disease Recognition Dataset/Original Images/Original Images"
BASE_DIR = "/kaggle/working/banana_split"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

# ---------------------------
# Create train/val split
# ---------------------------
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

for cls in os.listdir(DATA_DIR):
    cls_path = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(cls_path):
        continue
    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)
    
    files = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
    random.shuffle(files)
    split_idx = int(0.8 * len(files))
    
    for f in files[:split_idx]:
        shutil.copy(os.path.join(cls_path, f), os.path.join(TRAIN_DIR, cls, f))
    for f in files[split_idx:]:
        shutil.copy(os.path.join(cls_path, f), os.path.join(VAL_DIR, cls, f))

print("✅ Dataset successfully split into train & val folders")

# ---------------------------
# Parameters
# ---------------------------
IMG_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 30

# ---------------------------
# Data Generators
# ---------------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb"
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb"
)

# ---------------------------
# Build model - DenseNet121
# ---------------------------
num_classes = train_generator.num_classes

base_model = DenseNet121(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

base_model.trainable = False  # Freeze initially

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.4)(x)
output = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------
# Callbacks
# ---------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1)
lr_reduce = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1)

# ---------------------------
# Train
# ---------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, lr_reduce]
)

# ---------------------------
# Save class names & model
# ---------------------------
np.save("class_names.npy", np.array(list(train_generator.class_indices.keys())))
model.save("banana_disease_densenet121.keras")
print("✅ Training complete. Model saved as 'banana_disease_densenet121.keras'")
