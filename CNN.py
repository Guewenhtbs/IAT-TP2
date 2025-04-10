import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import csv
import os


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Hyperparam
learning_rate = 0.0005
ks1 = 3
ks2 = 3
ks3 = 5
epoch = 10
nbCouches = 3

output_dir = "./vplots/v-plots"

# Créer le dossier 'plots' s'il n'existe pas (chemin relatif)
os.makedirs(output_dir, exist_ok=True)

# Charger et préparer les données
train_dir = '/home/madmuses/Documents/IAT/vehicle dataset/train'
val_dir = '/home/madmuses/Documents/IAT/vehicle dataset/test'
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    train_dir, target_size=(64, 64), batch_size=32,
    class_mode='binary')
val_generator = datagen.flow_from_directory(
    val_dir, target_size=(64, 64), batch_size=32,
    class_mode='binary')

# Définir le modèle CNN
model = Sequential([
    Conv2D(32, (ks1,ks1), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (ks2,ks2), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (ks3,ks3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
]) 

# Compilation du modèle
model.compile(
    optimizer = keras.optimizers.Adam(learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy'])



# Entraînement du modèle
history = model.fit(train_generator, validation_data=val_generator, epochs=epoch)

# Évaluation
loss, accuracy = model.evaluate(val_generator)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Visualisation des courbes de loss et d'accuracy
plt.figure(figsize=(12, 4))

# Courbe de perte (loss)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Courbe de perte')
plt.xlabel('Épochs')
plt.ylabel('Loss')
plt.legend()

# Courbe de précision (accuracy)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Courbe de précision')
plt.xlabel('Épochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()

# Save plot
filename = f"{output_dir}/run_l-{learning_rate}_ksize-{ks1}.{ks2}.{ks3}_e-{epoch}_n-{nbCouches}__acc-{accuracy}_loss-{loss}.png"
plt.savefig(filename)
print(f"Saved under : {filename}")

# Ajouter les paramètres et résultats dans un fichier CSV
csv_file = "results.csv"
file_exists = os.path.isfile(csv_file)

# Écrire les données dans le fichier CSV
with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Écrire l'en-tête si le fichier est nouveau
    #if not file_exists:
    writer.writerow(["learning_rate", "ks1","ks2","ks3", "epoch", "nbCouches", "accuracy", "loss", "plot_filename"])
    # Ajouter une ligne avec les valeurs actuelles
    writer.writerow([learning_rate, ks1, ks2, ks3, epoch, nbCouches, accuracy, loss, filename])

print(f"Résultats ajoutés au fichier CSV : {csv_file}")