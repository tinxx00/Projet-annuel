from ultralytics import YOLO

# 📍 Chemin vers le fichier YAML du dataset
dataset_path = 'dataset/data.yaml'

# 📦 Charger un modèle pré-entraîné YOLOv8n (nano)
model = YOLO('yolov8n.pt')  # tu peux essayer aussi 'yolov8s.pt' si tu veux un modèle plus grand

# 🏋️ Lancer l'entraînement
model.train(
    data=dataset_path,
    epochs=30,
    imgsz=416,
    batch=16,
    name='yolov8_dent',
    project='runs/detect'
)
