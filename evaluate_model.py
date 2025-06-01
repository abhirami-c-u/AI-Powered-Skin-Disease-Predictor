import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model_path = 'skin_classifier/skin_model.keras'
model = load_model(model_path)
print(f"Loaded model from {model_path}")

# Load metadata
df = pd.read_csv('skin_disease_data/HAM10000_metadata.csv')

def find_image_path(image_id):
    p1 = os.path.join("skin_disease_data", "HAM10000_images_part_1", image_id + ".jpg")
    p2 = os.path.join("skin_disease_data", "HAM10000_images_part_2", image_id + ".jpg")
    return p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)

df['image_path'] = df['image_id'].apply(find_image_path)
df = df[df['image_path'].notnull()]

# Keep only top 4 classes
top_classes = df['dx'].value_counts().nlargest(4).index.tolist()
df = df[df['dx'].isin(top_classes)]

# Validation split
from sklearn.model_selection import train_test_split
_, val_df = train_test_split(df, test_size=0.2, stratify=df['dx'], random_state=42)

# Data generator
val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col='image_path',
    y_col='dx',
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

# Predictions
val_gen.reset()
preds = model.predict(val_gen)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes
labels = list(val_gen.class_indices.keys())


# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"\nOverall Accuracy: {acc * 100:.2f}%")


# Actual vs Predicted Counts
actual_counts = pd.Series(y_true).value_counts().sort_index()
predicted_counts = pd.Series(y_pred).value_counts().sort_index()

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, actual_counts, width, label='Actual')
plt.bar(x + width/2, predicted_counts, width, label='Predicted')
plt.xticks(x, labels)
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Actual vs Predicted Counts per Class')
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted_counts.png")
plt.show()
plt.close() 
# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
