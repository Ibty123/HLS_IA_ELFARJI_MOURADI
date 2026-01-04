# P5_CNN_Acceleration_HLS : Accélération d'une Intelligence Artificielle Embarquée (LeNet CNN) sur Zynq via HLS

**Ibtissam EL FARJI -
Malak MOURADI**

**Encadrant :** S. Bilavarn

**Date :**  04 Janvier 2026


## Objectif du projet

L’objectif est de réaliser le flot complet :
1. **Référence logicielle** (inférence sur ARM Cortex-A9)
2. **Conversion float → fixed-point** (Q8 sur 16 bits)
3. **Génération d’un accélérateur HLS** (IP dans la logique programmable)
4. **Déploiement sur ZedBoard**  

Le projet compare trois versions :
- **SW** : exécution purement CPU (référence)
- **HW_SEQ** : accélération HLS sans pragmas d’optimisation 
- **HW_PAR** : accélération HLS optimisée (parallélisation via pragmas) 

---

## Architecture du réseau (LeNet-5 adapté MNIST)

- Input : 28×28×1  
- Conv1 : 20 filtres 5×5 → 24×24×20  
- Pool1 : MaxPool 2×2 (stride 2) → 12×12×20  
- Conv2 : 40 filtres 5×5 → 8×8×40  
- Pool2 : MaxPool 2×2 (stride 2) → 4×4×40  
- FC1 : 640 → 400  
- FC2 : 400 → 10  
- Output : Softmax (10 classes) 

---

## Fixed-point (Q8 / 16 bits)

Pour rendre le modèle synthétisable efficacement :
- Format **Q8** sur **short (16 bits)** : 1 bit signe, 7 bits entier, 8 bits fractionnaires
- Conversion : `x_fixed = round(x * 2^8)`
- Les poids/biais sont intégrés **en dur** dans `Weights.h` (tableaux `static short const`) 

---

## Outils / Technologies

- **TensorFlow / Keras** : entraînement et export des poids (HDF5)
- **C/C++** : implémentation LeNet (référence + fixed-point)
- **Vivado HLS** : synthèse HLS de la top-function
- **SDSoC / Vitis** : intégration système, génération AXI/DMA, build `.elf`
- **ZedBoard** : exécution Linux embarqué (BOOT.BIN, image.ub) :contentReference

---


## Compilation / Génération accélérateur (HLS + SDSoC / Vitis)

Cette section décrit les étapes générales pour compiler, synthétiser et intégrer l’accélérateur matériel basé sur HLS.

### 1) Version Software (SW)

- Compiler l’application en mode **software only** (sans accélération matérielle).
- Exécuter l’inférence LeNet-5 sur le processeur **ARM Cortex-A9**.
- Cette version sert de **référence fonctionnelle et temporelle**.

### 2) Version Hardware Séquentielle (HW_SEQ)

- Identifier la fonction top-level du CNN (ex : `lenet_cnn_fixed`).
- Marquer cette fonction comme **accélérateur matériel** dans SDSoC / Vitis.
- Ne pas ajouter de pragmas d’optimisation HLS.
- Générer automatiquement :
  - l’IP HLS,
  - les interfaces **AXI**,
  - le contrôleur **DMA** pour les transferts PS <-> PL.
- Compiler et générer l’exécutable final.

### 3) Version Hardware Parallèle (HW_PAR)

- Ajouter des **pragmas HLS** dans le code :
  - `#pragma HLS PIPELINE`
  - `#pragma HLS UNROLL`
  - `#pragma HLS ARRAY_PARTITION`
- Objectif :
  - réduire le nombre de cycles,
  - augmenter le parallélisme interne,
  - améliorer les performances globales.
- Lancer la synthèse HLS et l’intégration système.
- Générer l’exécutable optimisé.

---


## Résultats expérimentaux

Les tests ont été réalisés sur la carte **ZedBoard (Zynq-7000)** en utilisant le jeu de données **MNIST**.  
Toutes les versions du système utilisent les mêmes poids entraînés et convertis en **fixed-point Q8**, garantissant une comparaison équitable.

| Version   | Accuracy (%) | Temps total (s) | Cycles HW |
|-----------|--------------|-----------------|-----------|
| SW        | 97.74        | ~2400           | N/A       |
| HW_SEQ   | 97.74        | ~2700           | 6 447 885 |
| HW_PAR   | 97.74        | ~300            | 750 235   |

### Analyse des résultats

- La version **SW** (logicielle) sert de référence fonctionnelle et temporelle.
- La version **HW_SEQ** présente un temps d’exécution plus élevé que la version SW :
  - ceci est dû principalement à l’overhead de communication entre le **Processing System (PS)** et la **Programmable Logic (PL)** via AXI/DMA.
- La version **HW_PAR** montre une amélioration significative :
  - réduction drastique du nombre de cycles,
  - accélération globale malgré l’overhead de transfert,
  - validation de l’efficacité des pragmas HLS et du parallélisme matériel.

---


## Auteurs


- **Ibtissam EL FARJI**
- **Malak MOURADI**

Encadré par : **Sébastien Bilavarn**


