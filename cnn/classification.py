import os
import cv2
import shutil
import random
import pandas as pd
from tqdm import tqdm
import albumentations as A

# ===============================
# 🔹 SECTION 1 : DÉCOUPE ET CLASSEMENT DES IMAGES
# ===============================

base_images_path = r"C:\Users\tinhi\Downloads\dental annotation.v16-augmented-rcnn.tensorflow\train"
annotations_path = r"C:\Users\tinhi\Downloads\dental annotation.v16-augmented-rcnn.tensorflow\train\_annotations.csv"
output_dir = r"C:\Users\tinhi\Downloads\classification_data"

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(annotations_path)

print("📂 Découpage et classement des images...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(base_images_path, row['filename'])
    label = row['class']
    class_dir = os.path.join(output_dir, label)
    os.makedirs(class_dir, exist_ok=True)

    image = cv2.imread(img_path)
    if image is None:
        continue

    xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
    crop = image[ymin:ymax, xmin:xmax]
    if crop.size == 0:
        continue

    output_path = os.path.join(class_dir, f"{idx}_{row['filename']}")
    cv2.imwrite(output_path, crop)

print(f"✅ Images classées dans : {output_dir}")

# ===============================
# 🔹 SECTION 2 : AUGMENTATION DES CLASSES SOUS-REPRÉSENTÉES
# ===============================

print("\n🔁 Démarrage de l'augmentation des classes...")

target_count = 600
classes_to_augment = ["Badly Decayed", "Normal", "Overhang", "Restoration"]

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.GaussianBlur(p=0.3),
    A.Resize(128, 128)
])

def augment_and_save(image_path, save_dir, aug_id):
    image = cv2.imread(image_path)
    if image is None:
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)['image']
    save_path = os.path.join(save_dir, f"aug_{aug_id}_" + os.path.basename(image_path))
    cv2.imwrite(save_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

for class_name in classes_to_augment:
    class_path = os.path.join(output_dir, class_name)
    if not os.path.isdir(class_path):
        print(f"❌ Classe introuvable : {class_name}")
        continue

    image_files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    current_count = len(image_files)

    print(f"\n🔍 Classe '{class_name}': {current_count} images")

    if current_count >= target_count:
        print("✅ Déjà assez d'images.")
        continue

    print(f"🚀 Augmentation en cours pour atteindre {target_count} images...")

    i = 0
    while current_count < target_count:
        img_file = random.choice(image_files)
        img_path = os.path.join(class_path, img_file)
        augment_and_save(img_path, class_path, i)
        i += 1
        current_count += 1

    print(f"✅ Classe '{class_name}' après augmentation : {current_count} images")

print("\n🎉 Préparation terminée : images classées et équilibrées.")
