"""
ETRS606 - TP1 : Problème MNIST - Version Optimisée pour IA Embarquée (STM32)
Auteur : [Ton Nom/Pseudo]

Ce script entraîne un réseau de neurones denses (Multi-Layer Perceptron) sur le jeu de données MNIST[cite: 4].
Il contient des images en niveaux de gris de chiffres manuscrits (0 à 9)[cite: 5].
L'architecture a été pensée pour minimiser l'empreinte mémoire (nombre de synapses) 
et la complexité des calculs afin d'être soutenable sur un microcontrôleur contraint (ex: STM32).
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def main():
    # ---------------------------------------------------------
    # 1. Chargement et préparation des données
    # ---------------------------------------------------------
    print("Chargement des données MNIST...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Chaque image est de taille 28x28 pixels[cite: 6].
    # Aplatissement des images : on passe d'une matrice 28x28 à un vecteur de 784 pixels[cite: 10, 14].
    # Normalisation : on ramène la valeur des pixels entre 0 et 1 (division par 255.0).
    x_train = x_train.reshape(-1, 784) / 255.0
    x_test = x_test.reshape(-1, 784) / 255.0

    # Note : Nous n'utilisons PAS le "one-hot encoding" sur les labels (y_train/y_test).
    # Ils restent sous forme d'entiers (0 à 9) pour économiser de la RAM lors du traitement, 
    # ce qui est la pratique recommandée avec la fonction coût "sparse_categorical_crossentropy"[cite: 44].

    # ---------------------------------------------------------
    # 2. Création du modèle "Embarqué" (MLP)
    # ---------------------------------------------------------
    model = Sequential([
        # Couche d'entrée implicite de dimension 784[cite: 17].
        # Couche cachée : 32 neurones (au lieu de 128) pour limiter drastiquement la dimension de W (nombre de paramètres).
        # Activation ReLU : calcul très rapide (simple condition logique) idéal pour un processeur limité.
        Dense(32, activation='relu', input_shape=(784,)),
        
        # Couche de sortie dense de 10 neurones, produisant une distribution de probabilité sur les 10 classes[cite: 18].
        Dense(10, activation='softmax')
    ])

    print("\n--- Architecture du modèle optimisé ---")
    model.summary()
    # Le modèle ne possède qu'environ 25 000 paramètres (soit ~100 Ko), 
    # ce qui rentre parfaitement dans la mémoire Flash d'un STM32.

    # ---------------------------------------------------------
    # 3. Compilation du modèle
    # ---------------------------------------------------------
    # Optimiseur : Adam (moderne, combine momentum + RMSprop, converge vite)[cite: 40].
    # Fonction coût : sparse_categorical_crossentropy (pratique pour MNIST si les chiffres sont codés 0-9)[cite: 44].
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # ---------------------------------------------------------
    # 4. Entraînement du modèle
    # ---------------------------------------------------------
    print("\n--- Début de l'entraînement ---")
    history = model.fit(x_train, y_train, 
                        epochs=10, 
                        batch_size=128, # On groupe par lots pour accélérer l'entraînement sur le PC
                        validation_data=(x_test, y_test))

    # ---------------------------------------------------------
    # 5. Évaluation finale
    # ---------------------------------------------------------
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n--- Résultats finaux ---")
    print(f"Précision sur les données de test : {accuracy * 100:.2f}%")
    print("Le modèle est prêt à être exporté (ex: TensorFlow Lite for Microcontrollers) pour le STM32 !")

if __name__ == "__main__":
    main()
