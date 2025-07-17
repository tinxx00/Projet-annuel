import os
import math
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from yolo.yolomini.model import YOLOMini
from yolo.yolomini.dataset import YoloDataset, custom_collate_fn
from yolo.yolomini.loss import yolo_loss

from tqdm import tqdm

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = YoloDataset(
        images_dir=r"C:\Users\tinhi\Downloads\dental annotation.v16-augmented-rcnn.yolov8 (1)\train\images",
        labels_dir=r"C:\Users\tinhi\Downloads\dental annotation.v16-augmented-rcnn.yolov8 (1)\train\labels",
        transform=transform
    )

    val_dataset = YoloDataset(
        images_dir=r"C:\Users\tinhi\Downloads\dental annotation.v16-augmented-rcnn.yolov8 (1)\valid\images",
        labels_dir=r"C:\Users\tinhi\Downloads\dental annotation.v16-augmented-rcnn.yolov8 (1)\valid\labels",
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=custom_collate_fn)

    model = YOLOMini(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    model.train()
    os.makedirs("runs", exist_ok=True)

    best_val_loss = float("inf")
    patience = 5
    counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(10):
        model.train()
        train_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"🔵 Entraînement Epoch {epoch+1}"):
            images = images.to(device)
            targets = [t.to(device) if isinstance(t, torch.Tensor) else t for t in targets]

            if torch.isnan(images).any():
                print("❌ NaN détecté dans les images, batch ignoré.")
                continue

            outputs = model(images)

            if isinstance(outputs, torch.Tensor):
                raise ValueError("Le modèle doit retourner un tuple (bbox_pred, conf_pred, class_pred)")

            bbox_pred, conf_pred, class_pred = outputs

            if torch.isnan(bbox_pred).any() or torch.isnan(conf_pred).any() or torch.isnan(class_pred).any():
                print("❌ NaN détecté dans les prédictions, batch ignoré.")
                continue

            loss = yolo_loss((bbox_pred, conf_pred, class_pred), targets, num_classes=10)

            if isinstance(loss, torch.Tensor):
                loss_value = loss.item()
            else:
                loss_value = loss

            if math.isnan(loss_value):
                print("⚠️ NaN dans la loss d'entraînement, batch ignoré.")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss_value

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"📘 Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"🟣 Validation Epoch {epoch+1}"):
                images = images.to(device)
                targets = [t.to(device) if isinstance(t, torch.Tensor) else t for t in targets]

                if torch.isnan(images).any():
                    print("❌ NaN détecté dans les images de validation, batch ignoré.")
                    continue

                outputs = model(images)
                bbox_pred, conf_pred, class_pred = outputs

                if torch.isnan(bbox_pred).any() or torch.isnan(conf_pred).any() or torch.isnan(class_pred).any():
                    print("❌ NaN détecté dans les prédictions validation, batch ignoré.")
                    continue

                loss = yolo_loss((bbox_pred, conf_pred, class_pred), targets, num_classes=10)

                if isinstance(loss, torch.Tensor):
                    loss_value = loss.item()
                else:
                    loss_value = loss

                if math.isnan(loss_value):
                    print("⚠️ NaN dans la loss de validation, batch ignoré.")
                    continue

                val_loss += loss_value

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"🟪 Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), "runs/best_weights.pth")
            print(f"✅ Nouveau meilleur modèle sauvegardé avec loss = {best_val_loss:.4f}")
        else:
            counter += 1
            if counter >= patience:
                print("⛔ Early stopping déclenché.")
                break

    # Courbes de perte
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Courbes de perte YOLOMini')
    plt.legend()
    plt.savefig("runs/loss_curve.png")

# Point d’entrée
if __name__ == "__main__":
    train()
