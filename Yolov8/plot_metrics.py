import pandas as pd
import matplotlib.pyplot as plt
import os

# 📁 Chemin vers le fichier CSV
csv_path = 'runs/detect/yolov8_dent/results.csv'
save_dir = 'figures'
os.makedirs(save_dir, exist_ok=True)

# 📄 Charger les données
df = pd.read_csv(csv_path)

# 🎨 Tracer les losses
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train/box_loss'], label='train_box_loss')
plt.plot(df['epoch'], df['train/cls_loss'], label='train_cls_loss')
plt.plot(df['epoch'], df['train/dfl_loss'], label='train_dfl_loss')
plt.plot(df['epoch'], df['val/box_loss'], label='val_box_loss')
plt.plot(df['epoch'], df['val/cls_loss'], label='val_cls_loss')
plt.plot(df['epoch'], df['val/dfl_loss'], label='val_dfl_loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Évolution des pertes (Losses) - YOLOv8")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_dir}/loss_curve.png")
#plt.show()

# (Optionnel) 🎯 Courbe mAP50
if 'metrics/mAP_0.5' in df.columns:
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['metrics/mAP_0.5'], label='mAP@0.5')
    plt.plot(df['epoch'], df['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95')
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Évolution des mAP - YOLOv8")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/map_curve.png")
    #plt.show()
