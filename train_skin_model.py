import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load metadata
df = pd.read_csv('skin_disease_data/HAM10000_metadata.csv')

# Find image paths
def find_image_path(image_id):
    p1 = os.path.join("skin_disease_data", "HAM10000_images_part_1", image_id + ".jpg")
    p2 = os.path.join("skin_disease_data", "HAM10000_images_part_2", image_id + ".jpg")
    return p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)

df['image_path'] = df['image_id'].apply(find_image_path)
df = df[df['image_path'].notnull()]

# Select top 4 largest classes dynamically
top_classes = df['dx'].value_counts().nlargest(4).index.tolist()
df = df[df['dx'].isin(top_classes)]

# Train-validation split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_gen = datagen.flow_from_dataframe(train_df, x_col='image_path', y_col='dx',
                                        target_size=(224, 224), class_mode='categorical', batch_size=32)

val_gen = datagen.flow_from_dataframe(val_df, x_col='image_path', y_col='dx',
                                      target_size=(224, 224), class_mode='categorical', batch_size=32)

# Compute class weights
classes_sorted = np.array(sorted(train_df['dx'].unique()))
class_weights_array = compute_class_weight(class_weight='balanced', classes=classes_sorted, y=train_df['dx'])
class_weights = {train_gen.class_indices[cls]: weight for cls, weight in zip(classes_sorted, class_weights_array)}

# Build model
base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(len(classes_sorted), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint('skin_classifier/best_model.keras', save_best_only=True, monitor='val_loss')
]

# Train initial frozen base
model.fit(train_gen, validation_data=val_gen, epochs=15, class_weight=class_weights, callbacks=callbacks)

# Fine-tune last 20 layers
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=10, class_weight=class_weights, callbacks=callbacks)

# Save final model
model.save('skin_classifier/last_skin_model.keras')
print("Final model saved!")

# Evaluate
val_gen.reset()
preds = model.predict(val_gen)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes
labels = list(train_gen.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=labels))
