from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
model = load_model('last_skin_model.keras')

classes = ['nv', 'bkl', 'bcc', 'mel']
  

UPLOAD_FOLDER = 'static/uploaded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Disease descriptions (fill or expand as needed)
disease_descriptions = {

    'NV': 'Melanocytic nevi (NV) are common benign skin moles.',
    'BKL': 'Benign keratosis-like lesions are non-cancerous skin growths.',
    'MEL': 'Melanoma is a serious form of skin cancer that can spread to other parts of the body.',
    'BCC': 'Basal cell carcinoma is a type of skin cancer that begins in the basal cells.'
}

# Actions to be taken for each disease (fill or expand as needed)
disease_actions = {
    
    'NV': 'Generally harmless, but monitor for any changes and consult a dermatologist if needed.',
    'BKL': 'Usually harmless but monitor any changes and consult if needed.',
    'MEL': 'Seek immediate medical attention for biopsy and treatment.',
    'BCC': 'See a healthcare professional for biopsy and treatment.'
}

def get_risk_assessment(disease, confidence):
    # Example logic — you can adjust thresholds & messages
    if disease == 'MEL':
        if confidence > 70:
            return 'High Risk', 'Please consult a doctor immediately.'
        elif confidence > 40:
            return 'Moderate Risk', 'Consider scheduling a checkup soon.'
        else:
            return 'Low Risk', 'Monitor your skin regularly.'
    else:
        # For other diseases, general risk messaging
        if confidence > 70:
            return 'Likely', 'Follow recommended actions and consult a doctor.'
        elif confidence > 40:
            return 'Possible', 'Be cautious and consider a doctor visit.'
        else:
            return 'Unlikely', 'Keep monitoring and maintain skin care.'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('fullname', 'User').strip() or 'User'
    consent = request.form.get('agreement')

    
    file = request.files['skinphoto']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = load_img(path, target_size=(224, 224))
    img_arr = img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    prediction = model.predict(img_arr)[0]
    idx = np.argmax(prediction)
    result = classes[idx]
    confidence = prediction[idx] * 100

    image_name = file.filename
    result_upper = result.upper()
    description = disease_descriptions.get(result_upper, "Description not available.")
    actions = disease_actions.get(result_upper, "Actions not available.")
    risk_level, risk_message = get_risk_assessment(result_upper, confidence)

    return render_template('result.html',
                           name=name,
                           result=result_upper,
                           confidence=round(confidence, 2),
                           image=path,
                           image_name=image_name,
                           description=description,
                           actions=actions,
                           risk_level=risk_level,
                           risk_message=risk_message)

if __name__ == '__main__':
    app.run(debug=True)
