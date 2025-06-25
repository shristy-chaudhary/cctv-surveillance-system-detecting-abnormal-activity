# Video Action Recognition for Surveillance Systems

This repository contains a PyTorch implementation for video action recognition, primarily focused on identifying abnormal or suspicious activities in surveillance footage. The project leverages various 3D Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) architectures to learn spatio-temporal features from video sequences.

## 🌟 Features

* **Custom Video Dataset Loader**: Efficiently loads video frames from `.mp4` files, handles variable frame counts, and prepares sequences for model input.
* **Multiple Model Architectures**: Implements and supports various state-of-the-art models for video action recognition, including:
    * **3DCNN + LSTM**: A hybrid approach combining 3D convolutions for spatial feature extraction per frame, followed by an LSTM for temporal sequence modeling. (Your initial approach)
    * **C3D**: Pure 3D Convolutional Neural Networks for direct spatio-temporal feature learning.
    * **R(2+1)D**: Decomposes 3D convolutions into 2D spatial and 1D temporal operations for improved efficiency and performance.
    * **SlowFast Networks**: Utilizes two pathways (slow for spatial, fast for temporal) to capture different aspects of video dynamics.
    * **I3D (Inflated 3D ConvNets)**: Inflates 2D image classification architectures (like Inception) into 3D for video understanding.
    * (I have used the above two.)
* **Modular Code**: Organized into functions and classes for clarity and reusability.
* **Training & Evaluation Loop**: Includes standard PyTorch training and evaluation procedures with progress tracking (`tqdm`) and learning rate scheduling.
* **Google Colab Ready**: Designed for easy setup and execution in Google Colab environments, including direct dataset download via KaggleHub.

## 🚀 Getting Started

Follow these steps to set up and run the project in Google Colab.

### 1. Google Colab Setup

Open a new Google Colab notebook and ensure you have a GPU runtime enabled (`Runtime > Change runtime type > GPU`).

### 2. Install Dependencies

Run the following cell to install all required Python packages:

```python
!pip install -q opencv-python numpy torch torchvision scikit-learn tqdm kagglehub

Dataset Link
1.https://www.kaggle.com/datasets/mateohervas/dcsass-dataset
2.https://www.kaggle.com/datasets/minhajuddinmeraj/anomalydetectiondatasetucf
