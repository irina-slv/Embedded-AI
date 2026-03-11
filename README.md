# IA Embarqué 
Projet IA Embarqué L3_SPI des "Ny" 


# ETRS606 - TP1 : Multi-Layer Perceptron (MLP) sur MNIST pour IA Embarquée [cite: 1, 3]


L'objectif est d'implémenter et d'optimiser un réseau de neurones denses (MLP) pour résoudre le problème de classification d'images MNIST, avec pour contrainte finale un déploiement sur une cible matérielle limitée (microcontrôleur STM32).

## 1. Description du Problème et des Données
Le dataset MNIST contient des images en niveaux de gris de chiffres manuscrits (de 0 à 9).
**Entrée :** Chaque image fait $28\times28$ pixels.Pour notre MLP, elle est aplatie en un vecteur 1D de 784 pixels. L'information de position spatiale n'est donc pas conservée.
* **Sortie :** 10 classes possibles (chiffres 0 à 9). Le modèle produit une distribution de probabilité sur ces 10 classes.

## 2. Partie 1 : Étude des Architectures et Fonctions d'Activation 
Nous avons testé quatre architectures avec l'optimiseur Adam et la fonction de coût `categorical_crossentropy` sur 10 epochs. L'ensemble $W$ représente les poids/synapses du modèle.

| Modèle | Couches Cachées | Fonction d'Activation | Précision (Val Acc) | Paramètres (Taille W) |
| :--- | :--- | :--- | :--- | :--- |
| **A** | 0 | Softmax | ~ 92.36 % | 7 850 |
| **B** | 2 (128, 64) | ReLU | ~ 97.76 % | 109 386 |
| **C** | 1 (128) | Tanh | ~ 97.83 % | 101 770 |
| **D** | 1 (128) | Sigmoid | *Plus lent à converger* | 101 770 |

**Analyse des compromis (Accuracy vs. Mémoire):**
Bien que les modèles avec couches cachées (B et C) atteignent d'excellentes précisions (> 97%), ils requièrent plus de 100 000 paramètres. 
Pour un déploiement sur un **STM32**, la mémoire Flash et la RAM sont très limitées. Le calcul d'exponentielles complexes requises par `tanh` ou `sigmoid` est également coûteux en énergie et en cycles d'horloge. 

**Choix de l'architecture pour l'embarqué :** Nous avons opté pour un compromis : un modèle **ReLU** avec une seule petite couche cachée de **32 neurones**. ReLU permet un calcul très rapide (simple condition logique) idéal pour un microcontrôleur, tout en réduisant drastiquement le nombre de paramètres par rapport à un modèle à 128 neurones.

## [3. Partie 2 : Choix de l'Algorithme d'Optimisation 
En utilisant notre architecture légère (ReLU + Sortie Softmax), nous avons comparé différents optimiseurs sur 10 epochs.

* **Adam :** 96.76 % (Convergence rapide, combine momentum et RMSprop)
* **RMSprop :** 96.68 % (Très stable, adapte le taux d'apprentissage)
* **SGD :** 94.34 % (Classique, intuition de base, mais plus lent à converger)
* **Adagrad :** 89.66 % (Baisse son taux d'apprentissage trop agressivement ici)

**Conclusion :** `Adam` et `RMSprop` offrent les meilleures performances de convergence rapide pour cette architecture.

## 4. Partie 3 : Choix de la Fonction Coût
Le choix de la fonction de coût dépend de la sortie du modèle et du format des labels.
* **`categorical_crossentropy` :** Utilisée initialement, requiert d'encoder les labels en "one-hot" (ex: `[0, 0, 1, ...]` pour le chiffre 2).
* **`sparse_categorical_crossentropy` :** Permet d'utiliser directement les entiers (ex: `2`).

**Conclusion :** L'entraînement avec `sparse_categorical_crossentropy` donne des résultats identiques (~96.84% d'accuracy) mais évite la création de grands vecteurs d'encodage "one-hot" en mémoire. C'est une optimisation très pertinente pour réduire l'empreinte RAM lors de la préparation des données sur un système contraint.

