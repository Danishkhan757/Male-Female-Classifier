import streamlit as ui
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Page configuration
ui.set_page_config(page_title="Male/Female Classifier", page_icon="👤", layout="centered")

ui.title("👤 Male or Female Image Classifier")
ui.write("Image upload kijiye aur AI batayega ki wo Male hai ya Female.")

# 1. Load your trained model
# Apne model ka sahi naam/path yahan likhein (e.g., 'male_female_model.h5')
@ui.cache_resource
def load_my_model():
    try:
        model = tf.keras.models.load_model('male_female_model.h5')
        return model
    except Exception as e:
        ui.error(f"Model load karne me dikkat aayi: {e}")
        return None

model = load_my_model()

# 2. Image Upload
uploaded_file = ui.file_uploader("Koi bhi image select karein...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    ui.image(image, caption='Uploaded Image', use_column_width=True)
    ui.write("")
    ui.write("### Brainstorming... (Predicting)")

    if model is None:
        ui.warning("Model file ('male_female_model.h5') missing hai! Please check karein.")
    else:
        # 3. Image Preprocessing
        # (Apne model ke hisab se size 224x224 ya 150x150 change kar sakte hain)
        size = (224, 224) 
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image)
        
        # Normalize/Scale image if required by your model
        img_reshape = img_array / 255.0  
        img_reshape = np.expand_dims(img_reshape, axis=0)

        # 4. Prediction
        prediction = model.predict(img_reshape)
        
        # Maan lete hain: 0 = Female, 1 = Male (Aap apne model ke hisab se badal sakte hain)
        if prediction[0][0] > 0.5:
            ui.success(f"Prediction: **Male** (Confidence: {prediction[0][0]*100:.2f}%)")
        else:
            ui.error(f"Prediction: **Female** (Confidence: {(1 - prediction[0][0])*100:.2f}%)")