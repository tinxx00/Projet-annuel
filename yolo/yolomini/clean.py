import os

LABELS_DIR = "C:/Users/tinhi/Downloads/dental annotation.v16-augmented-rcnn.yolov8 (1)/train/labels"

for file_name in os.listdir(LABELS_DIR):
    if file_name.endswith(".txt"):
        path = os.path.join(LABELS_DIR, file_name)
        with open(path, 'r') as file:
            lines = file.readlines()

        cleaned_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cleaned_line = " ".join(parts[:5])  # garde uniquement les 5 premières valeurs
                cleaned_lines.append(cleaned_line + "\n")

        # Remplacer le fichier avec les lignes nettoyées
        with open(path, 'w') as file:
            file.writelines(cleaned_lines)

print("✅ Tous les fichiers ont été nettoyés.")
