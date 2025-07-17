from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import os
from ultralytics import YOLO

app = FastAPI()

# 📍 Dossier de base
BASE_DIR = os.path.dirname(__file__)

# 🧠 Chargement des modèles
cnn_model_path = os.path.join(BASE_DIR, 'dentaI_model.h5')
cnn_model = tf.keras.models.load_model(cnn_model_path, compile=False)

yolo_model_path = os.path.join(BASE_DIR, 'best.pt')
yolo_model = YOLO(yolo_model_path)

# 🏷️ Classes du modèle CNN
class_names = ['Abcess', 'Badly Decayed', 'Caries', 'Crown', 'Normal', 'Overhang', 'Post', 'RCT', 'Restoration']

# 🧼 Prétraitement pour CNN
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((128, 128))  # adapte à la taille d'entraînement
    image = image.convert("RGB")
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# 🧠 Endpoint CNN
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        input_data = preprocess_image(image)
        prediction = cnn_model.predict(input_data)[0]

        predicted_index = int(np.argmax(prediction))
        predicted_label = class_names[predicted_index]
        confidence = float(prediction[predicted_index])

        return JSONResponse({
            "filename": file.filename,
            "prediction": predicted_label,
            "confidence": round(confidence, 3)
        })

    except UnidentifiedImageError:
        return JSONResponse({"error": "Le fichier n'est pas une image valide."}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# 🧠 Endpoint YOLOv8
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        results = yolo_model.predict(image)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = yolo_model.names[cls_id]
                detections.append({
                    "label": label,
                    "confidence": round(conf, 3)
                })

        return JSONResponse({
            "filename": file.filename,
            "detections": detections
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
