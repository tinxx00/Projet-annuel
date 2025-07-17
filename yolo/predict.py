import torch
from yolomini.model import YOLOMini  # ton modèle YOLO
import cv2
import matplotlib.pyplot as plt

# Charger le modèle
model = YOLOMini(num_classes=9)
model.load_state_dict(torch.load("runs/weights.pth", map_location="cpu"))
model.eval()

# Charger une image
img_path = "/yolo/data\\R.webp"

image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized = cv2.resize(image_rgb, (128, 128))
input_tensor = torch.from_numpy(resized / 255.).permute(2, 0, 1).unsqueeze(0).float()

# Prédiction
with torch.no_grad():
    output = model(input_tensor)

# Visualisation simple
plt.imshow(image_rgb)
plt.title("Prédiction YOLO (non annotée)")
plt.axis("off")
plt.show()
