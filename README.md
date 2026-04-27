# 🤖 ETRS606 - IA Embarquée | Projet NYA

> **Projet IA Embarquée L3_SPI — Équipe "Ny"**  
> Université Savoie Mont Blanc  
> Carte cible : **STM32 NUCLEO-N657X0** + shield capteurs **X-NUCLEO-IKS01A3**

---

## 📋 Vue d'ensemble

Ce dépôt couvre l'ensemble du module ETRS606 : de l'entraînement d'un réseau de neurones en Python jusqu'au déploiement Edge AI sur microcontrôleur STM32, en passant par la connectivité IoT cloud via ThingSpeak.

```
TP1 → MLP/MNIST (Python/TensorFlow)
TP2 → Capteurs I2C sur STM32 (C/HAL)
TP3 → Connectivité Cloud ThingSpeak (NetX Duo / ThreadX)
TP4 → Cloud vs Edge AI (X-CUBE-AI + ONNX)
TP5 → Examen de compétences (démo + pitch)
```

---

## 🗂️ Structure du dépôt

```
ETRS606-IA-Embarquee/
├── TP1_MNIST/
│   ├── mnist_mlp.py              # Entraînement MLP
│   ├── mnist_softmax_model/      # Modèle sauvegardé Keras
│   └── mnist_softmax.onnx        # Export ONNX pour MATLAB/STM32
├── TP2_Capteurs/
│   └── Nx_TCP_Echo_Client/       # Projet STM32CubeIDE
│       └── FSBL/
│           ├── Core/Src/main.c
│           └── NetXDuo/App/app_netxduo.c
├── TP3_Cloud/
│   └── thingspeak_analysis.m     # Script MATLAB analyse données
├── TP4_EdgeAI/
│   └── meteostat_model.py        # Modèle météo + export ONNX
└── README.md
```

---

## TP1 — Multi-Layer Perceptron sur MNIST

### Objectif
Implémenter et optimiser un réseau de neurones dense (MLP) pour la classification d'images MNIST, avec contrainte de déploiement sur microcontrôleur STM32.

### Dataset
- **784 entrées** (images 28×28 aplaties)
- **10 sorties** (chiffres 0–9)
- 60 000 images d'entraînement / 10 000 de test

### Résultats — Comparaison des fonctions d'activation

| Modèle | Couches cachées | Activation | Val Accuracy | Paramètres |
|:---|:---|:---|:---:|:---:|
| A | 0 | Softmax | ~92.4% | 7 850 |
| B | 2 (128, 64) | ReLU | ~97.8% | 109 386 |
| C | 1 (128) | Tanh | ~97.8% | 101 770 |
| D | 1 (128) | Sigmoid | ~97.1% | 101 770 |
| **E** *(embarqué)* | **1 (32)** | **ReLU** | **~96.8%** | **~26 000** |

> **Choix pour l'embarqué :** ReLU + 1 couche de 32 neurones.  
> ReLU = `max(0, x)` → calcul quasi-instantané sur MCU, pas d'exponentielle.

### Résultats — Comparaison des optimiseurs

| Optimiseur | Accuracy | Remarque |
|:---|:---:|:---|
| Adam | 96.76% | Meilleur compromis vitesse/précision |
| RMSprop | 96.68% | Très stable |
| SGD | 94.34% | Convergence lente mais intuitive |
| Adagrad | 89.66% | Taux d'apprentissage trop agressif |

### Fonction coût recommandée pour MNIST

```python
# Si labels one-hot [0,0,1,0,...] :
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Si labels entiers 0-9 (recommandé pour économiser la RAM) :
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
```

### Export vers l'embarqué (ONNX)

```python
import tf2onnx
import tensorflow as tf

model = tf.keras.models.load_model("mnist_softmax_model")
spec = (tf.TensorSpec((None, 784), tf.float32, name="input"),)
tf2onnx.convert.from_keras(model, input_signature=spec, output_path="mnist_softmax.onnx")
```

---

## TP2 — Interface Capteurs sur STM32

### Matériel
- **NUCLEO-N657X0** : Cortex-M55 @ 600 MHz, NPU Neural-ART 600 GOPS
- **X-NUCLEO-IKS01A3** : capteurs MEMS sur bus I2C

| Capteur | Grandeur | Driver |
|:---|:---|:---|
| HTS221 | Température + Humidité | `hts221_reg.c/h` |
| LPS22HH | Pression atmosphérique | `lps22hh_reg.c/h` |
| LSM6DSO16IS | Accélération 3 axes | `lsm6dso16is_reg.c/h` |

### Architecture du projet STM32

```
Nx_TCP_Echo_Client/FSBL/
├── Core/Src/
│   └── main.c          ← Init capteurs I2C + lancement ThreadX
├── NetXDuo/App/
│   └── app_netxduo.c   ← DHCP + lecture capteurs + envoi ThingSpeak
└── NetXDuo/Target/
    └── nx_stm32_eth_config.h
```

> ⚠️ **Point critique architecture STM32N657** : Ce MCU utilise un schéma **FSBL** (First Stage Boot Loader). Le code applicatif tourne dans le projet FSBL lui-même avec ThreadX (Azure RTOS). `MX_ThreadX_Init()` ne retourne jamais — tout le code applicatif doit être dans un thread.

### Fonctions bas-niveau I2C (obligatoires pour les drivers ST)

```c
/* À implémenter dans main.c */
int32_t platform_write(void *handle, uint8_t reg, uint8_t *bufp, uint16_t len) {
    sensor_handle_t *s = (sensor_handle_t*)handle;
    uint8_t r = reg;
    /* HTS221 requiert le bit d'auto-incrément (MSB) */
    if (s->dev_addr == HTS221_I2C_ADDRESS) r |= 0x80;
    if (HAL_I2C_Mem_Write(s->handle, s->dev_addr, r,
                          I2C_MEMADD_SIZE_8BIT, bufp, len, 1000) == HAL_OK)
        return 0;
    return -1;
}
```

### Indicateurs LED

| LED | État | Signification |
|:---|:---|:---|
| 🟢 Vert | `GPIO_PIN_RESET` (active low) | Disponible, attente lecture |
| 🔴 Rouge | `GPIO_PIN_RESET` (active low) | Lecture capteurs en cours |

> Sur la NUCLEO-N657X0, les LEDs sont **actives à l'état bas** (`GPIO_PIN_RESET` = LED allumée).

---

## TP3 — Connectivité Cloud ThingSpeak

### Configuration du canal ThingSpeak

1. Créer un compte sur [thingspeak.com](https://thingspeak.com)
2. **New Channel** → configurer les champs :

| Field | Grandeur | Unité |
|:---|:---|:---|
| field1 | Température | °C |
| field2 | Pression | hPa |
| field3 | Accélération X | mg |
| field4 | Accélération Y | mg |
| field5 | Accélération Z | mg |

3. Récupérer la **Write API Key** dans l'onglet "API Keys"

### Envoi HTTP GET via NetX Duo

La fonction `thingspeak_send()` dans `app_netxduo.c` :

```c
/* Requête HTTP GET vers api.thingspeak.com */
snprintf(request, sizeof(request),
         "GET /update?api_key=%s&field1=%.2f&field2=%.2f"
         "&field3=%.2f&field4=%.2f&field5=%.2f"
         " HTTP/1.1\r\nHost: api.thingspeak.com\r\n"
         "Connection: close\r\n\r\n",
         THINGSPEAK_WRITE_KEY, temp_c, press_hpa, ax, ay, az);
```

> ⚠️ **Limite API ThingSpeak gratuite** : minimum **15 secondes** entre deux envois.  
> Pour la soutenabilité écologique, nous avons choisi **120 secondes** (2 minutes).

### Bug connu — callback DHCP vide

Le projet d'exemple `Nx_TCP_Echo_Client` de STMicroelectronics laisse `ip_address_change_notify_callback()` **vide**, ce qui bloque le thread indéfiniment après le DHCP. Le correctif obligatoire :

```c
static VOID ip_address_change_notify_callback(NX_IP *ip_instance, VOID *ptr)
{
    ULONG ip_addr, mask;
    nx_ip_address_get(ip_instance, &ip_addr, &mask);
    printf("[NET] IP : %lu.%lu.%lu.%lu\r\n",
           (ip_addr>>24)&0xFF, (ip_addr>>16)&0xFF,
           (ip_addr>>8)&0xFF,  ip_addr&0xFF);
    tx_semaphore_put(&DHCPSemaphore);  /* ← indispensable ! */
}
```

### printf via LPUART1

Après un Generate Code CubeMX, l'UART peut changer de nom (`huart1` → `hlpuart1`). Vérifier :

```c
/* Dans main.c — doit pointer vers le bon handle */
PUTCHAR_PROTOTYPE {
    HAL_UART_Transmit(&hlpuart1, (uint8_t *)&ch, 1, 0xFFFF);
    return ch;
}
```

### Analyse MATLAB (TP3b)

```matlab
% Récupérer les données depuis ThingSpeak et calculer la moyenne par minute
data = thingSpeakRead(CHANNEL_ID, 'Fields', [1 2], ...
                      'NumPoints', 100, 'ReadKey', 'READ_API_KEY');
temp_moy = mean(data(:,1), 'omitnan');
pres_moy = mean(data(:,2), 'omitnan');
fprintf("Temp moy : %.2f°C | Pression moy : %.2f hPa\n", temp_moy, pres_moy);
```

---

## TP4 — Cloud vs Edge AI

### Comparaison des approches

| Critère | Cloud AI (ThingSpeak) | Edge AI (STM32) |
|:---|:---:|:---:|
| Latence | Élevée (réseau) | Très faible (local) |
| Consommation réseau | Importante | Nulle |
| Puissance de calcul | Illimitée | Contrainte MCU |
| Indépendance réseau | ❌ | ✅ |
| Mise à jour modèle | Facile | Reflash nécessaire |

### Pipeline de déploiement du modèle météo

```
Python/TensorFlow  →  ONNX  →  MATLAB (Cloud)
                          ↓
                     X-CUBE-AI  →  Code C  →  STM32
```

#### Export Python → ONNX

```python
import tf2onnx
spec = (tf.TensorSpec((None, N_FEATURES), tf.float32, name="input"),)
tf2onnx.convert.from_keras(model, input_signature=spec,
                            output_path="meteostat_model.onnx")
```

#### Import ONNX → MATLAB

```matlab
model = importONNXNetwork("meteostat_model.onnx", OutputLayerType="classification");
scores = predict(model, input_data);
[prob, class_idx] = max(scores);
```

#### Import ONNX → STM32 via X-CUBE-AI

1. CubeMX → **X-CUBE-AI** → importer le `.onnx`
2. Analyser l'empreinte mémoire (RAM Flash)
3. Générer le code C → intégrer dans le projet FSBL

---

## 🔧 Guide de démarrage rapide (pour les futurs étudiants)

### Prérequis logiciels

| Outil | Version recommandée | Usage |
|:---|:---|:---|
| STM32CubeIDE | 1.16+ | Développement C embarqué |
| Python | 3.9+ | Entraînement modèles IA |
| TensorFlow | 2.x | Framework IA |
| tf2onnx | latest | Export modèle |
| MATLAB | R2023+ | Analyse cloud + inférence |

### Installation Python

```bash
pip install tensorflow tf2onnx meteostat pandas numpy matplotlib
```

### Pièges courants et solutions

#### 1. Projet FSBL vs AppliNonSecure
Le STM32N657X0 a une architecture multi-projets dans CubeIDE. Tout le code applicatif (capteurs, réseau) doit être dans le projet **FSBL** si vous utilisez le projet `Nx_TCP_Echo_Client`. Ne pas confondre avec l'architecture AppliNonSecure/AppliSecure d'un projet CubeMX généré from scratch.

#### 2. Makefile avec caractères spéciaux
Le caractère `é` dans les noms de dossiers (`IAEmbarqué`) casse le makefile. **Utiliser uniquement des caractères ASCII** dans les chemins de projet : `IAEmbarque`.

#### 3. Double définition de `nx_app_thread_entry`
L'exemple `Nx_TCP_Echo_Client` contient une version vide de `nx_app_thread_entry`. En ajoutant votre propre version dans `USER CODE BEGIN 1`, vous obtenez une redéfinition. **Supprimer l'originale vide**, ne garder que la vôtre.

#### 4. ThreadX ne retourne pas
`MX_ThreadX_Init()` appelle `tx_kernel_enter()` qui ne retourne jamais. Tout code après cette ligne est mort. La logique applicative **doit être dans un thread** créé dans `MX_NetXDuo_Init()`.

#### 5. DHCP sémaphore jamais libéré
Voir section TP3 — le callback `ip_address_change_notify_callback` doit appeler `tx_semaphore_put()`.

#### 6. printf muet après Generate Code
Vérifier que `PUTCHAR_PROTOTYPE` pointe vers `hlpuart1` et non `huart1` si CubeMX a généré LPUART1.

#### 7. HTS221 — calibration obligatoire
La température brute du HTS221 est inutilisable sans calibration. Lire les 4 points de calibration **une seule fois au démarrage** :
```c
hts221_temp_deg_point_0_get(&ctx, &t0_degC);
hts221_temp_deg_point_1_get(&ctx, &t1_degC);
hts221_temp_adc_point_0_get(&ctx, &t0_out);
hts221_temp_adc_point_1_get(&ctx, &t1_out);
/* Conversion : */
temp_c = ((raw - t0_out) * (t1_degC - t0_degC) / (t1_out - t0_out)) + t0_degC;
```

---

## 🌱 Soutenabilité écologique (P_TEDS)

Ce projet intègre plusieurs choix techniques orientés réduction d'impact énergétique :

- **Fréquence d'envoi cloud réduite à 2 minutes** (120 s au lieu du minimum de 15 s), réduisant les connexions TCP de 75%
- **Utilisation de `tx_thread_sleep()`** entre les lectures : le CPU est suspendu, ThreadX cède les ressources aux autres tâches
- **Edge AI privilégié** pour les décisions temps-réel : pas de transfert réseau pour chaque inférence
- **ReLU choisi comme fonction d'activation** : calcul `max(0,x)` sans exponentielle, réduisant la consommation CPU vs sigmoid/tanh
- **Modèle compact** (1 couche, 32 neurones) pour minimiser l'empreinte Flash/RAM et les cycles d'inférence

---

## 📊 Résultats finaux du système complet

| Composant | Valeur mesurée |
|:---|:---|
| Température (HTS221) | ✅ Calibrée, affichée en °C |
| Pression (LPS22HH) | ✅ Affichée en hPa |
| Accélération (LSM6DSO) | ✅ X/Y/Z en mg |
| Envoi ThingSpeak | ✅ HTTP GET, toutes les 5 min |
| Latence DHCP | ~2–5 s au démarrage |
| Console série | ✅ LPUART1 @ 115200 baud |

---

## 👥 Équipe

Projet réalisé par l'équipe **"Ny"** — L3 SPI  
Université Savoie Mont Blanc — ETRS606 IA Embarquée

---

## 📄 Licence

Code source sous licence MIT.  
Drivers STMicroelectronics sous leurs licences respectives (voir headers des fichiers).
