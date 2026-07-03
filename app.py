from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications.efficientnet import preprocess_input 

import numpy as np
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image
import requests
import uuid
from io import BytesIO

app = Flask(__name__)

# --- Configuration ---
# 1. Get the directory where this app.py file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Define the exact file name (Make sure your file matches this exactly!)
MODEL_FILENAME = "EfficientNetB0_ODIR_OfflineAug.h5"

# 3. Create the full absolute path
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

CONF_THRESHOLD = 0.45 # Standard low confidence threshold for known retinal images
INVALID_IMAGE_THRESHOLD = 0.35 # Threshold to reject clearly non-retinal images

# --- Load model ---
model = None
print(f"DEBUG: Attempting to load model from: {MODEL_PATH}")

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"File not found at {MODEL_PATH}")
        
    model = load_model(MODEL_PATH)
    print("SUCCESS: Model loaded successfully!")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model.")
    print(f"   Reason: {e}")
    print(f"   SOLUTION: Ensure '{MODEL_FILENAME}' is in the folder: {BASE_DIR}")

# --- Class labels (exact order used in training) ---
class_labels = ['Retinal Vein Occlusion', 'ageDegeneration', 'cataract', 'diabetes', 'myopia', 'normal']

# --- DISEASE INFORMATION ---
DISEASE_INFO = {
    "Retinal Vein Occlusion": {
        "display_name": "Retinal Vein Occlusion (RVO)",
        "image_folder_name": "Retinal Vein Occlusion", 
        "tagline": "A serious vascular blockage requiring urgent attention.",
        "description": "RVO is the blockage of small veins that carry blood away from the retina. This leads to blood and fluid leakage, causing a rapid and often severe loss of vision. It is a critical risk, especially for those with high blood pressure or diabetes.",
        "symptoms": "Sudden, painless blurring or loss of vision, often described as a dark shadow or blind spot.",
        "treatment": "Intravitreal injections (e.g., anti-VEGF), laser photocoagulation, and strict management of underlying systemic conditions like hypertension."
    },
    "ageDegeneration": {
        "display_name": "Age-related Macular Degeneration (AMD)",
        "image_folder_name": "ageDegeneration",
        "tagline": "The leading cause of vision loss in older adults.",
        "description": "AMD causes damage to the macula—the central part of the retina responsible for sharp, detailed central vision. It progresses in two forms: dry (gradual) and wet (rapid leakage/bleeding).",
        "symptoms": "Blurred or 'wavy' central vision, dark, blank spots, and difficulty recognizing faces or reading fine print.",
        "treatment": "For dry AMD: high-dose antioxidant and mineral supplements (AREDS). For wet AMD: regular anti-VEGF injections to stop new blood vessel growth."
    },
    "cataract": {
        "display_name": "Cataract",
        "image_folder_name": "cataract",
        "tagline": "Clouding of the eye's lens, easily treatable.",
        "description": "A cataract is a clouding of the normally clear lens of the eye, which eventually obstructs the passage of light, leading to blurry vision. While common with age, it is highly treatable.",
        "symptoms": "Hazy or blurred vision, colors appearing faded, poor night vision, and increased sensitivity to glare/lights.",
        "treatment": "Surgical removal of the cloudy lens and replacement with an artificial intraocular lens (IOL) is highly effective."
    },
    "diabetes": {
        "display_name": "Diabetic Retinopathy (DR)",
        "image_folder_name": "diabetes",
        "tagline": "Damage to retinal vessels caused by high blood sugar.",
        "description": "Diabetic Retinopathy is a complication of diabetes that damages the blood vessels in the light-sensitive tissue at the back of the eye (retina). It is a progressive condition that can lead to irreversible blindness if not managed.",
        "symptoms": "Floaters, blurred vision, impaired color vision, and areas of missing or dark vision.",
        "treatment": "Strict blood sugar and blood pressure control. Advanced treatments include anti-VEGF injections, steroids, and vitrectomy surgery for severe cases."
    },
    "myopia": {
        "display_name": "Pathologic Myopia (High Nearsightedness)",
        "image_folder_name": "myopia",
        "tagline": "Severe nearsightedness posing retinal detachment risk.",
        "description": "Pathologic Myopia is a severe form of nearsightedness where the eyeball stretches too much. This extreme stretching thins and damages the retina, increasing the risk of complications like retinal detachment, macular degeneration, and glaucoma.",
        "symptoms": "Extremely poor distant vision, severe distortion, and visual field loss.",
        "treatment": "Correction with glasses or contacts. Monitoring and surgical intervention (e.g., laser) to address secondary complications like retinal tears or holes."
    },
    "normal": {
        "display_name": "Healthy Retina",
        "image_folder_name": "normal",
        "tagline": "Clear vision and optimal retinal health.",
        "description": "This diagnosis indicates a healthy fundus (retina) without visible signs of the common diseases monitored by this screening tool. Regular checkups are still vital for long-term preventative care.",
        "symptoms": "Clear, stable vision and absence of visual disturbances.",
        "treatment": "Maintain regular comprehensive eye examinations, especially after age 40, and manage overall health (diet, exercise, blood pressure)."
    }
}

# --- Helpers ---
def preprocess_for_model(img_path_or_data, img_size=224):
    """Load and preprocess image for EfficientNet"""
    if isinstance(img_path_or_data, str):
        img = Image.open(img_path_or_data).convert("RGB")
    else:
        img = Image.open(img_path_or_data).convert("RGB")

    img = img.resize((img_size, img_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def predict_topk(x, k=3):
    if model is None:
        raise ValueError("Model not loaded.")
    preds = model.predict(x)[0]
    top_idx = preds.argsort()[::-1][:k]
    return [(class_labels[i], float(preds[i])) for i in top_idx]

# --- Routes ---

# 1. LANDING PAGE
@app.route('/', methods=['GET'])
def landing_page():
    return render_template('landing.html', DISEASE_INFO=DISEASE_INFO)

# 2. PREDICTION PAGE (Upload Form)
@app.route('/predict', methods=['GET'])
def index():
    if model is None:
        return render_template('index.html', model_error=True)
    return render_template('index.html', model_error=False)

@app.route('/static/uploads/<path:filename>')
def uploaded_file(filename):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.isfile(file_path):
        abort(404)
    return send_from_directory(UPLOAD_DIR, filename)

# 3. PREDICTION EXECUTION ROUTE
@app.route('/predict', methods=['POST'])
def predict_post():
    if model is None:
        return redirect(url_for('index'))

    url = request.form.get('image_url')
    file = request.files.get('file')

    filename = None

    if url and url.strip():
        headers = {
             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            image_data = BytesIO(response.content)

            filename = f"url_image_{uuid.uuid4()}.jpg"
            save_path = os.path.join(UPLOAD_DIR, filename)

            image_data.seek(0)
            with Image.open(image_data) as img:
                 img.save(save_path, 'JPEG')

            image_data.seek(0)
            x = preprocess_for_model(image_data, img_size=224)

        except requests.exceptions.RequestException as e:
            return render_template('index.html', error_message=f"Error downloading image from URL. Check URL or network: {e}")
        except Exception as e:
            return render_template('index.html', error_message=f"The downloaded file is not a valid image or another URL error occurred: {e}")

    elif file and file.filename:
        try:
            filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
            save_path = os.path.join(UPLOAD_DIR, filename)
            file.save(save_path)
            x = preprocess_for_model(save_path, img_size=224)
        except Exception as e:
            return render_template('index.html', error_message="The uploaded file could not be processed. Not a valid image.")
    else:
        return render_template('index.html', error_message="You must upload a file OR provide an image URL.")

    # --- PREDICTION ---
    try:
        top3 = predict_topk(x, k=3)
        top1_label, top1_prob = top3[0][0], top3[0][1]
    except Exception as e:
        return render_template('index.html', error_message=f"Prediction failed: {e}")

    # --- RESULT DISPLAY LOGIC ---

    # 1. REJECTION CHECK: If confidence is below 50%, reject the image entirely.
    if top1_prob < INVALID_IMAGE_THRESHOLD:
        result_data = {
            'filename': filename,
            'is_invalid_image': True, # Flag for the template
            'analysis_date': datetime.now().strftime('%d %b %Y'),
            'analysis_time': datetime.now().strftime('%I:%M %p').lstrip('0'),
            'model_used': 'EfficientNetB0',
        }
        return render_template('results.html', **result_data)

    # 2. NORMAL FLOW: Handle low/high confidence retinal images.
    result_data = {
        'filename': filename,
        'top3': top3,
        'top1_label': top1_label,
        'top1_prob': top1_prob,
        'is_low_conf': top1_prob < CONF_THRESHOLD,
        'disease_info': DISEASE_INFO.get(top1_label, {}),
        'analysis_date': datetime.now().strftime('%d %b %Y'),
        'analysis_time': datetime.now().strftime('%I:%M %p').lstrip('0'),
        'model_used': 'EfficientNetB0',
    }

    return render_template('results.html', **result_data)


if __name__ == "__main__":
    app.run(debug=True)