# MNIST Handwriting Recognition with CNN

A computer vision project that classifies handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset. 

## 🧠 Overview

This project demonstrates how to build a neural network using TensorFlow and Keras to:

- Load and visualize the MNIST dataset
- Normalize and preprocess image data
- Construct a CNN using convolutional and pooling layers
- Train the model to classify grayscale images of digits
- Evaluate performance using test accuracy
- Visualize predictions and intermediate data

---

## 📁 Dataset

**Source**: [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

- **Images**: 28x28 grayscale
- **Classes**: Digits from 0 to 9 (10 total classes)
- **Training set**: 60,000 images
- **Test set**: 10,000 images

---

## 🏗️ CNN Architecture

The CNN model was implemented using Keras Sequential API and contains:

- **Input Layer**: `(28, 28, 1)` grayscale images
- **4 Convolutional Layers**:
  - Filters: 16, 32, 64, 128
  - Kernel Size: 3
  - Batch Normalization
  - ReLU Activation
- **Global Average Pooling Layer**
- **Dense Output Layer**: 10 neurons (one per digit class)

---

## 📊 Model Performance

| Metric     | Value      |
|------------|------------|
| **Loss**   | 0.2012     |
| **Accuracy** | 93.5%     |

The model achieves strong accuracy on the test set, demonstrating its ability to generalize to unseen handwriting.

---

## 📷 Visualizations

### 🔍 Sample Image from Training Set

![sample_image](./training_data_example.png)

### 📈 Sample Prediction Output

![predictions](./predictions_on_test_set.png)

---

## ⚙️ Setup

### 🧪 Requirements

- Python 3.x
- TensorFlow / Keras
- NumPy
- Seaborn
- Matplotlib

