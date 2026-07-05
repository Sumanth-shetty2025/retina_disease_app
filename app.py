import gc
import logging
import os
import tempfile
import traceback
import uuid
from datetime import datetime

# Keep TensorFlow conservative on small Render instances.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import numpy as np
import requests
import tensorflow as tf
from flask import Flask, abort, redirect, render_template, request, send_from_directory, url_for
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    # TensorFlow threading can only be configured before runtime initialization.
    logger.info("TensorFlow threading was already initialized before configuration.")

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
IMAGE_SIZE = 224
MODEL_USED = "EfficientNetB0"
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

Image.MAX_IMAGE_PIXELS = 20_000_000

# --- Load model ---
model = None
logger.info("Attempting to load model from: %s", MODEL_PATH)

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"File not found at {MODEL_PATH}")

    model = load_model(MODEL_PATH, compile=False)
    logger.info("Model loaded successfully.")
except Exception as exc:
    logger.exception("Critical error: could not load model.")
    print(f"CRITICAL ERROR: Could not load model.")
    print(f"   Reason: {exc}")
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
def log_exception(message):
    logger.exception(message)
    traceback.print_exc()


def preprocess_pil_image(img, img_size=IMAGE_SIZE):
    """Convert a PIL image into one EfficientNet batch."""
    converted_img = None
    if img.mode != "RGB":
        converted_img = img.convert("RGB")
        img = converted_img

    try:
        resized = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32)
        batch = arr[np.newaxis, ...]
        batch = preprocess_input(batch)

        del resized
        del arr
        return batch
    finally:
        if converted_img is not None:
            converted_img.close()


def preprocess_image_file(image_path, img_size=IMAGE_SIZE):
    """Open, validate, close, and preprocess an image from disk."""
    try:
        with Image.open(image_path) as img:
            return preprocess_pil_image(img, img_size=img_size)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file could not be processed. Not a valid image.") from exc


def remove_file_safely(file_path):
    if not file_path:
        return
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except OSError:
        logger.warning("Could not remove temporary file: %s", file_path, exc_info=True)


def stream_url_to_temp_file(url):
    temp_path = None
    bytes_read = 0

    try:
        with requests.get(url, headers=REQUEST_HEADERS, timeout=15, stream=True) as response:
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_DIR, suffix=".download") as temp_file:
                temp_path = temp_file.name
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue

                    bytes_read += len(chunk)
                    if bytes_read > MAX_DOWNLOAD_BYTES:
                        raise ValueError("The image URL is too large. Please use an image under 10 MB.")

                    temp_file.write(chunk)

        return temp_path
    except Exception:
        remove_file_safely(temp_path)
        raise


def save_url_image_and_preprocess(url):
    temp_path = None
    filename = f"url_image_{uuid.uuid4()}.jpg"
    save_path = os.path.join(UPLOAD_DIR, filename)

    try:
        temp_path = stream_url_to_temp_file(url)
        with Image.open(temp_path) as img:
            with img.convert("RGB") as rgb_img:
                rgb_img.save(save_path, "JPEG", quality=90, optimize=True)
                batch = preprocess_pil_image(rgb_img)

        return filename, batch
    except requests.exceptions.RequestException as exc:
        remove_file_safely(save_path)
        raise RuntimeError(f"Error downloading image from URL. Check URL or network: {exc}") from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        remove_file_safely(save_path)
        raise ValueError(f"The downloaded file is not a valid image or another URL error occurred: {exc}") from exc
    finally:
        remove_file_safely(temp_path)


def save_upload_and_preprocess(file):
    filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    save_path = os.path.join(UPLOAD_DIR, filename)

    try:
        file.save(save_path)
        batch = preprocess_image_file(save_path)
        return filename, batch
    except Exception:
        remove_file_safely(save_path)
        raise


def predict_topk(x, k=3):
    if model is None:
        raise ValueError("Model not loaded.")

    preds_tensor = model(x, training=False)
    preds = np.asarray(preds_tensor.numpy()[0], dtype=np.float32)
    top_idx = np.argpartition(preds, -k)[-k:]
    top_idx = top_idx[np.argsort(preds[top_idx])[::-1]]
    return [(class_labels[i], float(preds[i])) for i in top_idx]


def get_analysis_timestamp():
    now = datetime.now()
    return now.strftime('%d %b %Y'), now.strftime('%I:%M %p').lstrip('0')


def base_result_data(filename):
    analysis_date, analysis_time = get_analysis_timestamp()
    return {
        'filename': filename,
        'analysis_date': analysis_date,
        'analysis_time': analysis_time,
        'model_used': MODEL_USED,
    }


def render_prediction_form_error(message):
    return render_template('index.html', error_message=message)

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
    x = None

    if url and url.strip():
        try:
            filename, x = save_url_image_and_preprocess(url.strip())
        except Exception as exc:
            log_exception("URL image processing failed.")
            return render_prediction_form_error(str(exc))

    elif file and file.filename:
        try:
            filename, x = save_upload_and_preprocess(file)
        except Exception:
            log_exception("Uploaded image processing failed.")
            return render_prediction_form_error("The uploaded file could not be processed. Not a valid image.")
    else:
        return render_prediction_form_error("You must upload a file OR provide an image URL.")

    # --- PREDICTION ---
    try:
        logger.info("Running prediction for image: %s", filename)
        top3 = predict_topk(x, k=3)
        top1_label, top1_prob = top3[0][0], top3[0][1]
        logger.info("Prediction complete for %s: %s (%.4f)", filename, top1_label, top1_prob)
    except Exception as exc:
        log_exception("Prediction failed.")
        return render_prediction_form_error(f"Prediction failed: {exc}")
    finally:
        del x
        gc.collect()

    # --- RESULT DISPLAY LOGIC ---

    # 1. REJECTION CHECK: If confidence is below 50%, reject the image entirely.
    if top1_prob < INVALID_IMAGE_THRESHOLD:
        result_data = base_result_data(filename)
        result_data.update({
            'is_invalid_image': True, # Flag for the template
        })
        return render_template('results.html', **result_data)

    # 2. NORMAL FLOW: Handle low/high confidence retinal images.
    result_data = base_result_data(filename)
    result_data.update({
        'top3': top3,
        'top1_label': top1_label,
        'top1_prob': top1_prob,
        'is_low_conf': top1_prob < CONF_THRESHOLD,
        'disease_info': DISEASE_INFO.get(top1_label, {}),
    })

    return render_template('results.html', **result_data)


if __name__ == "__main__":
    app.run(debug=True)
