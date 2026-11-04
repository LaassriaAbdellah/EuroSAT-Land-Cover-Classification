# 🛰️ EuroSAT Land Cover Classification

### Satellite Image Classification Using k-NN, Softmax Regression & Convolutional Neural Networks

A comparative Computer Vision project exploring three machine learning and deep learning approaches for satellite image classification using the **EuroSAT RGB dataset**.

The project evaluates how traditional distance-based classification, linear models, and convolutional neural networks perform on land cover recognition.

**Technologies:** Python · TensorFlow/Keras · Scikit-learn · NumPy · Matplotlib · Pillow · Google Colab

---

## 📌 Project Overview

Satellite image classification plays an important role in remote sensing, environmental monitoring, and land use analysis.

This project aims to classify satellite images into three land cover categories:

- 🌲 **Forest** — Forested and vegetation-covered areas
- 🏘️ **Residential** — Urban and residential areas
- 🌊 **River** — Rivers and surrounding waterway areas

The main objective is to compare three classification approaches and investigate the advantages of convolutional neural networks in extracting spatial features from satellite imagery.

## 📊 Dataset

The project uses the **EuroSAT RGB dataset**, based on Sentinel-2 satellite imagery.

**Dataset source:** [EuroSAT on Zenodo](https://zenodo.org/records/7711810)

| Property | Description |
|---|---|
| Dataset | EuroSAT RGB |
| Selected classes | Forest, Residential, River |
| Images per class | 300 |
| Total images | 900 |
| Original resolution | 64 × 64 pixels |
| Processed resolution | 32 × 32 pixels |
| Color channels | RGB (3 channels) |
| Random seed | 42 |

### Data Preprocessing

The preprocessing pipeline includes:

1. Loading and converting images to RGB.
2. Resizing images to 32 × 32 pixels.
3. Converting pixel values to `float32`.
4. Normalizing pixel intensities to `[0, 1]`.
5. Applying a stratified Train/Validation/Test split.

### Dataset Split

| Dataset | Percentage | Images |
|---|---:|---:|
| Training | 60% | 540 |
| Validation | 20% | 180 |
| Testing | 20% | 180 |
| **Total** | **100%** | **900** |

Each split preserves the same class distribution.

The test set is reserved for final evaluation and is not used for hyperparameter selection.

---

## 🧠 Models Implemented

### 1. k-Nearest Neighbors (k-NN)

A distance-based classification approach using flattened RGB images.

- Input representation: 3,072 features per image.
- Evaluated neighbors: `K = [1, 3, 5, 7, 9]`.
- Distance metrics: Manhattan (L1) and Euclidean (L2).
- Hyperparameter selection based on validation accuracy.

**Best configuration:**
- K = 1
- Distance = Manhattan (L1)
- Validation Accuracy = **81.11%**

### 2. Softmax Regression

A linear multiclass classifier implemented using TensorFlow/Keras.

**Architecture:**

`Input (3072) → Dense (3, Softmax)`

Training configuration:

- Optimizer: Adam
- Learning rate: 0.001
- Loss: Sparse Categorical Crossentropy
- Epochs: 30
- Batch size: 32
- Trainable parameters: 9,219

**Best Validation Accuracy: 82.78% (Epoch 21)**

The best model weights are selected using a validation-based checkpoint.

### 3. Convolutional Neural Network (CNN)

A convolutional architecture designed to learn spatial features directly from RGB images.

**Architecture:**

```text
Input (32 × 32 × 3)
        ↓
Conv2D (32 filters, 3×3, ReLU)
        ↓
MaxPooling2D (2×2)
        ↓
Conv2D (64 filters, 3×3, ReLU)
        ↓
MaxPooling2D (2×2)
        ↓
Flatten
        ↓
Dense (64, ReLU)
        ↓
Dense (3, Softmax)
```

Training configuration:

- Optimizer: Adam
- Learning rate: 0.001
- Loss: Sparse Categorical Crossentropy
- Epochs: 30
- Batch size: 32
- Trainable parameters: 167,107
- Model selection: `ModelCheckpoint` monitoring validation accuracy

**Best Validation Accuracy: 94.44% (Epoch 26)**

The checkpoint weights from epoch 26 are used for final test evaluation.

---

## 🏆 Experimental Results

All three models are evaluated on the same **180 previously unseen test images**.

### Test Set Performance

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-score (Macro) |
|---|---:|---:|---:|---:|
| k-NN | 73.33% | 0.7402 | 0.7333 | 0.7119 |
| Softmax | 72.78% | 0.7294 | 0.7278 | 0.7024 |
| **CNN** | **91.67%** | **0.9211** | **0.9167** | **0.9167** |

**The CNN achieves the highest performance**, correctly classifying 165 out of 180 test images.

It outperforms k-NN by approximately **18.33 percentage points** and Softmax Regression by **18.89 percentage points** in test accuracy.

### Class-Level Performance

Correct predictions out of 60 test images per class:

| Class | k-NN | Softmax | CNN |
|---|---:|---:|---:|
| Forest | 58/60 | 57/60 | 51/60 |
| Residential | 51/60 | 52/60 | 59/60 |
| River | 23/60 | 22/60 | 55/60 |

The CNN substantially improves recognition of Residential and River images.

However, k-NN and Softmax achieve better results on the Forest class, highlighting that the CNN does not outperform the other models on every individual class.

---

## 🔍 Key Findings

**1. Spatial features matter**

k-NN and Softmax operate on flattened images, while the CNN preserves spatial structure and learns local visual patterns through convolutional filters.

**2. CNNs perform better overall**

The CNN achieves substantially higher test accuracy and macro F1-score than both baseline models in this experiment.

**3. River classification is challenging for traditional approaches**

k-NN and Softmax frequently confuse River images with Forest or Residential areas.

**4. Classification errors remain**

The CNN makes 15 errors on the test set, including nine Forest images incorrectly classified as River.

These errors may be associated with visual similarities in color, texture, and spatial structure.

---

## 📈 Evaluation and Visualizations

The project includes:

- k-NN validation accuracy versus K for Manhattan and Euclidean distances.
- Softmax training and validation accuracy/loss curves.
- CNN training and validation accuracy/loss curves.
- Confusion matrices for all three models.
- Examples of correct and incorrect CNN predictions.
- Accuracy, macro Precision, macro Recall, and macro F1-score comparisons.

---

## 📁 Repository Contents

```text
EuroSAT-Land-Cover-Classification/
│
├── EuroSAT_Land_Cover_Classification.ipynb
├── Rapport_CV_EuroSat.pdf
└── README.md
```

The notebook contains the experimental workflow, while the PDF report provides a detailed explanation of the methodology, results, and discussion.

**Note:** The EuroSAT dataset is not included in the repository. It can be downloaded from the official Zenodo source.

---

## 🚀 How to Run

### Google Colab (Recommended)

1. Download or open the Jupyter notebook in Google Colab.
2. Connect to a Colab runtime.
3. Install any missing Python dependencies.
4. Download the EuroSAT RGB dataset from the provided source.
5. Configure the dataset path if necessary.
6. Execute the notebook cells in order.

The experiments were developed using Google Colab.

**Note:** If the runtime is restarted, temporary variables and files must be recreated or restored from saved artifacts.

---

## ⚠️ Limitations and Future Improvements

The experiments have several limitations:

- Only three of the available EuroSAT classes are considered.
- The dataset subset contains 900 images.
- Resizing to 32 × 32 pixels may remove fine visual details.
- The CNN is trained from scratch on a relatively small dataset.
- Results are based on a single stratified data split.

Potential improvements include:

- Increasing the number of training images.
- Extending classification to additional EuroSAT classes.
- Applying data augmentation.
- Exploring transfer learning with pretrained CNN architectures.
- Evaluating performance on additional independent data splits.

---

## 📄 Project Report

The full academic report is available in the repository:

📑 [View the EuroSAT Classification Report](Rapport_CV_EuroSat.pdf)

It includes the methodology, model architectures, experimental results, confusion matrices, error analysis, and conclusions.

---

## 🎓 Academic Context

**Program:** Master's in Information Systems and Intelligent Systems (M2SI)

**Institution:** Institut National de Statistique et d'Économie Appliquée (INSEA), Rabat, Morocco

**Course:** Computer Vision

**Academic Year:** 2026–2027

---

## 👤 Author

**Abdellah Laassria**

- [GitHub](https://github.com/LaassriaAbdellah)
- [LinkedIn](https://www.linkedin.com/in/abdellah-laassria-a450ba358/)

*Academic mini-project developed as part of the Computer Vision coursework at INSEA.*
