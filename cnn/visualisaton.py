import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# 🔄 Recharge le modèle
model = tf.keras.models.load_model("dentaI_model.h5", compile=False)

# 📂 Paramètres du jeu de données
img_size = (128, 128)
batch_size = 32
data_dir = r"C:\Users\tinhi\Downloads\classification_data"

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 🔮 Prédictions
y_true = val_gen.classes
y_pred_prob = model.predict(val_gen)
y_pred = np.argmax(y_pred_prob, axis=1)
class_names = list(val_gen.class_indices.keys())

# ✅ Matrice de confusion (définir cm AVANT de l'utiliser)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)

# 📊 Affichage avec seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Matrice de confusion")
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
#plt.show()


# 📃 Rapport détaillé
print("\nClassification report :\n")
print(classification_report(y_true, y_pred, target_names=class_names))
