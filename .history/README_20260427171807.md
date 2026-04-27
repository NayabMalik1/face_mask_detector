
# Face Mask Detection using CNN & Transfer Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange)](https://www.tensorflow.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the final project for the **Deep Learning** course. The goal is to implement and evaluate a Convolutional Neural Network (CNN) to classify images of people into three categories: wearing a mask correctly, **not** wearing a mask, and wearing a mask **incorrectly**.

## 📊 Key Results

The best model achieved a **96.0% test accuracy**, outperforming a simpler CNN and a pre-trained ResNet50 model.

| Model                   | Test Accuracy | Training Time (s) | Parameters |
|-------------------------|---------------|-------------------|------------|
| Original CNN (3 layers) | **96.0%**     | 738.10            | 3,305,027  |
| Simple CNN (1 layer)    | 94.8%         | 1181.49           | 8,129,667  |
| ResNet50 (Transfer Learning) | 78.2%     | 634.02            | 24,113,027 |

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Setup and Installation](#-setup-and-installation)
- [Models](#-models)
- [Training and Evaluation](#-training-and-evaluation)
- [Results](#-results)
- [License](#-license)

## 🎯 Project Overview
This project fulfills the requirements for Assignment No. 01 of the Deep Learning course. The tasks included:
- Preprocessing a real-world image dataset.
- Implementing a custom CNN from scratch.
- Training and evaluating the model, analyzing metrics for overfitting.
- Comparing the model's performance against simpler and more complex (ResNet50) architectures.

## 📁 Dataset
The [Face Mask Dataset](https://www.kaggle.com/datasets/shiekhburhan/face-mask-dataset) from Kaggle was used. It contains images in three classes:
1.  `with_mask`
2.  `without_mask`
3.  `incorrect_mask`

**Preprocessing steps:**
- Resized all images to `128x128` pixels.
- Normalized pixel values to the range `[0, 1]`.
- Split the data into training (70%), validation (15%), and test (15%) sets.
- Applied on-the-fly data augmentation (horizontal flipping) during training.

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NayabMalik1/face_mask_detector.git
    cd face_mask_detector
    ```

2.  **Install required packages:**
    ```bash
    pip install tensorflow==2.19.0 numpy matplotlib pandas scikit-learn kagglehub
    !pip install tensorflow
!pip install numpy
!pip install matplotlib
!pip install scikit-learn
!pip install pandas
!pip install kagglehub
!pip install pillow
    ```

3.  **Run the Notebook:** Open and execute all cells in the provided Jupyter Notebook (e.g., `face_mask_detection.ipynb`). The notebook will automatically download the dataset using the `kagglehub` library.

## 🧠 Models
Three different models were implemented and compared:
1.  **Original CNN:** A custom architecture with 3 convolutional layers, max-pooling, dropout for regularization, and a final dense layer for classification.
2.  **Simple CNN:** A baseline with only 1 convolutional layer followed by a dense layer.
3.  **ResNet50 (Transfer Learning):** A pre-trained ResNet50 model, used as a feature extractor (its base was frozen), with custom classification layers added on top.

## 📈 Training and Evaluation
- All models were trained for 20 epochs using the **Adam** optimizer and **Categorical Cross-Entropy** loss.
- **Callbacks** like `EarlyStopping` (patience=5) and `ModelCheckpoint` were used to prevent overfitting and save the best model.
- Performance was evaluated on the held-out test set using **accuracy, precision, recall, and F1-score**.

### 📉 Training Curves
Below are the accuracy and loss curves for the original CNN, showing consistent learning without significant overfitting.
*(**Replace this with your actual image path:** `![Original CNN Training Curves](accuracy_loss_curves.png)`)*

### 📊 Model Performance Comparison
The comparison of test accuracy, training time, and model size across the three architectures.
*(**Replace this with your actual image path:** `![Model Comparison](model_comparison.png)`)*

## 📊 Results
The detailed classification report for the best-performing model (Original CNN) is as follows:

| Class           | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| with_mask       | 0.94      | 0.95   | 0.95     | 719     |
| without_mask    | 0.94      | 0.97   | 0.95     | 710     |
| incorrect_mask  | 1.00      | 0.96   | 0.98     | 747     |

## 📄 License
This project is for educational purposes as part of the Deep Learning course at Fatima Jinnah Women University.

---
**Author:** Nayab Zahoor (2022-BSE-062)
