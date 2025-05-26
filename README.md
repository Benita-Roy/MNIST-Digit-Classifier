# MNIST Digit Classifier using TensorFlow/Keras

This project implements a simple feedforward neural network using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

## 📌 Overview

The model is a fully connected neural network (also known as a Multilayer Perceptron) trained on the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). The network is designed to flatten the 28x28 grayscale images into 784 input features and classify them into one of 10 classes.

## 🧠 Model Architecture

- **Input layer**: 784 features (28x28 image flattened)  
- **Hidden layer 1**: Dense layer with 120 neurons and ReLU activation  
- **Hidden layer 2**: Dense layer with 120 neurons and ReLU activation  
- **Output layer**: Dense layer with 10 neurons and softmax activation (for multi-class classification)

## 📊 Dataset

The MNIST dataset is automatically downloaded by Keras:

- 60,000 training samples  
- 10,000 test samples  
- Each image: 28x28 pixels (grayscale)

## 🧪 Training Details

- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  
- **Metric**: Accuracy  
- **Epochs**: 20  
- **Validation Split**: 10% of training data

## ✨ Results

The model achieves over **97% accuracy** on the MNIST test set after training for 20 epochs. This demonstrates the effectiveness of a simple dense neural network for image classification tasks.
