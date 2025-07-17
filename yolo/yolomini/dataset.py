import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class YoloDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.webp'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Nom du fichier image
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        label_path = os.path.join(
            self.labels_dir,
            img_filename.replace('.jpg', '.txt').replace('.png', '.txt').replace('.webp', '.txt')
        )

        # Chargement image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Chargement des labels
        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            boxes.append([x_center, y_center, width, height])
                            labels.append(class_id)
                        except ValueError:
                            print(f"⚠️ Ligne non convertible ignorée dans {label_path}: {line.strip()}")
                    else:
                        print(f"⚠️ Ligne ignorée (mauvais format) dans {label_path}: {line.strip()}")

        # Cas sans objet
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # Format YOLO : [class_id, x, y, w, h]
        targets = torch.cat([labels.unsqueeze(1).float(), boxes], dim=1)

        return image, targets

# 🔁 Collate function pour le DataLoader
def custom_collate_fn(batch):
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)  # (N, 5)

    images = torch.stack(images, dim=0)
    return images, targets
