# IA Embarqué 
Projet IA Embarqué L3_SPI

# Choix fonctions d'activation ( TP1 )
- Softmax
- ReLU
- Tanh
- Sigmoid





# ETRS606 - TP1 : Multi-Layer Perceptron (MLP) sur MNIST pour IA Embarquée [cite: 1, 3]


L'objectif est d'implémenter et d'optimiser un réseau de neurones denses (MLP) pour résoudre le problème de classification d'images MNIST[cite: 3, 15], avec pour contrainte finale un déploiement sur une cible matérielle limitée (microcontrôleur STM32).

## 1. Description du Problème et des Données
Le dataset MNIST contient des images en niveaux de gris de chiffres manuscrits (de 0 à 9)[cite: 5].
**Entrée :** Chaque image fait $28\times28$ pixels[cite: 6]. [cite_start]Pour notre MLP, elle est aplatie en un vecteur 1D de 784 pixels[cite: 6, 14]. L'information de position spatiale n'est donc pas conservée[cite: 14].
* **Sortie :** 10 classes possibles (chiffres 0 à 9)[cite: 7]. Le modèle produit une distribution de probabilité sur ces 10 classes[cite: 18, 20].

## 2. Partie 1 : Étude des Architectures et Fonctions d'Activation [cite: 25]
Nous avons testé quatre architectures avec l'optimiseur Adam et la fonction de coût `categorical_crossentropy` sur 10 epochs[cite: 29]. L'ensemble $W$ représente les poids/synapses du modèle[cite: 13].

| Modèle | Couches Cachées | Fonction d'Activation | Précision (Val Acc) | Paramètres (Taille W) |
| :--- | :--- | :--- | :--- | :--- |
| **A** | 0 | Softmax | ~ 92.36 % | 7 850 |
| **B** | 2 (128, 64) | ReLU | ~ 97.76 % | 109 386 |
| **C** | 1 (128) | Tanh | ~ 97.83 % | 101 770 |
| **D** | 1 (128) | Sigmoid | *Plus lent à converger* | 101 770 |

**Analyse des compromis (Accuracy vs. Mémoire)[cite: 37]:**
Bien que les modèles avec couches cachées (B et C) atteignent d'excellentes précisions (> 97%), ils requièrent plus de 100 000 paramètres[cite: 37]. 
Pour un déploiement sur un **STM32**, la mémoire Flash et la RAM sont très limitées. Le calcul d'exponentielles complexes requises par `tanh` ou `sigmoid` est également coûteux en énergie et en cycles d'horloge. 

**Choix de l'architecture pour l'embarqué :** Nous avons opté pour un compromis : un modèle **ReLU** avec une seule petite couche cachée de **32 neurones**. ReLU permet un calcul très rapide (simple condition logique) idéal pour un microcontrôleur, tout en réduisant drastiquement le nombre de paramètres par rapport à un modèle à 128 neurones.

## [3. Partie 2 : Choix de l'Algorithme d'Optimisation [cite: 38]
[cite_start]En utilisant notre architecture légère (Dense 32 ReLU + Sortie Softmax), nous avons comparé différents optimiseurs sur 10 epochs[cite: 39].

* **Adam :** 96.76 % (Convergence rapide, combine momentum et RMSprop [cite: 40])
* **RMSprop :** 96.68 % (Très stable, adapte le taux d'apprentissage [cite: 40])
* **SGD :** 94.34 % (Classique, intuition de base, mais plus lent à converger [cite: 40])
* **Adagrad :** 89.66 % (Baisse son taux d'apprentissage trop agressivement ici [cite: 40])

**Conclusion :** `Adam` et `RMSprop` offrent les meilleures performances de convergence rapide pour cette architecture.

## 4. Partie 3 : Choix de la Fonction Coût [cite: 41]
Le choix de la fonction de coût dépend de la sortie du modèle et du format des labels[cite: 42].
* **`categorical_crossentropy` :** Utilisée initialement, requiert d'encoder les labels en "one-hot" (ex: `[0, 0, 1, ...]` pour le chiffre 2)[cite: 44].
* **`sparse_categorical_crossentropy` :** Permet d'utiliser directement les entiers (ex: `2`)[cite: 44].

**Conclusion :** L'entraînement avec `sparse_categorical_crossentropy` donne des résultats identiques (~96.84% d'accuracy) mais évite la création de grands vecteurs d'encodage "one-hot" en mémoire[cite: 44]. C'est une optimisation très pertinente pour réduire l'empreinte RAM lors de la préparation des données sur un système contraint.

