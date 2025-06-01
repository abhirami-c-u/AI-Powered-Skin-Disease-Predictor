# Skin Disease Predictor - SkinSure AI

This project is an AI-based skin disease predictor system that leverages deep learning techniques to analyze and predict the skin conditions.

---

##  Features

- Preprocessing and data augmentation for robust training  
- Class imbalance handled with computed class weights  
- Transfer learning with MobileNetV2 pretrained on ImageNet  
- Fine-tuning of top layers for improved performance  
- Early stopping and model checkpointing to prevent overfitting  
 
---

##  Classes Used

This version uses the **4 most frequent classes** from the HAM10000 dataset:

- **nv** – Melanocytic nevi  
- **mel** – Melanoma  
- **bcc** – Basal cell carcinoma  
- **bkl** – Benign keratosis-like lesions  

---

##  Dataset

The project uses the [**HAM10000 Dataset**](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) (Kaggle).

**Note**: Only a subset of 4 classes is used for training to improve performance and balance.

- Download and extract the images and metadata.
- Images are stored in:  
  - `skin_disease_data/HAM10000_images_part_1`  
  - `skin_disease_data/HAM10000_images_part_2`

The `.csv` file `HAM10000_metadata.csv` contains the labels and metadata.

---

##  Model File

The trained Keras model is **not pushed** to GitHub (to keep the repo light and clean).

-  [Download skin_model.keras](https://drive.google.com/file/d/1iCenVIKIzp6iZLkUc-cKam2KCGJMAuNz/view?usp=drive_link)


---

##  Ignored Files (`.gitignore`)

The following files and folders are excluded from version control to keep the repo clean:

- Python cache files:  
  `__pycache__/`, `*.pyc`

- Model files:  
  `*.h5`, `*.keras`

- Dataset images and CSVs:  
  `*.jpg`, `*.jpeg`, `*.png`, `*.webp`, `*.csv`  
  `skin_disease_data/` folder is ignored

⚠️ Ensure you keep local copies of both the dataset and model.

---

##  Requirements

- Python 3.7+  
- TensorFlow 2.x  
- scikit-learn  
- pandas  
- numpy

Install dependencies with:

```bash
pip install tensorflow scikit-learn pandas numpy
```

---

## ⚠️ Disclaimer

This project is for educational and demonstration purposes only.  
It is **not intended for medical use** or professional diagnosis.  
Always consult a licensed dermatologist or healthcare provider for any concerns regarding skin conditions.

The predictions made by this AI system may not be accurate and should not be used as a substitute for clinical judgment.


