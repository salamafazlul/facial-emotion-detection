# Human Emotion Recognition from Images using CNN

## Project Summary
This project focuses on building a human emotion recognition model using Convolutional Neural Networks (CNN) to classify emotions from images. The model takes an image as input and predicts the emotion depicted in it. The project follows a structured approach that includes data preprocessing, model training, validation, and deployment as a web application.

## Project Overview
The human emotion recognition model classifies emotions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. The model is trained using a large dataset of labeled images and then deployed as a web application using Streamlit, allowing users to upload an image and receive emotion predictions in real-time.

## Project Steps

### 1. Demo & Functionality
- The model takes an image path as input.
- It predicts the emotion in the image.
- The predicted emotion is displayed along with the image.
- The dataset includes seven emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
- The accuracy is tested on a <a href="https://github.com/salamafazlul/facial-emotion-detection/tree/main/test_images"> sample dataset</a>.

### 2. Data Preprocessing
- The dataset consists of training and validation images, stored in respective folders.
- Each image is labeled based on its emotion.
- Libraries used:
  - TensorFlow
  - NumPy
  - OpenCV (cv2)
  - Matplotlib
  - Streamlit
- The dataset contains 28,000 images for training and 7,000 images for validation.
- Images are loaded into arrays and stored for model training.

### 3. Model Architecture
The CNN model is built using the Keras Sequential API with the following layers:
- **Convolutional Layers (Conv2D):** Extract features from images.
- **Batch Normalization:** Normalizes activations for stable training.
- **Max Pooling:** Reduces dimensionality while preserving important features.
- **Dropout Layers:** Prevents overfitting.
- **Dense Layers:** Fully connected layers for classification.
- **Softmax Activation:** Converts final layer outputs into probability distributions over seven emotions.
- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy
- **Evaluation Metric:** Accuracy

### 4. Model Training
- The model is trained using 28,000 training images and validated on 7,000 validation images.
- **Batch size:** 128
- **Epochs:** 50 (each epoch takes about 6 minutes to run)
- The trained model is saved to avoid retraining every time.

### 5. Model Deployment as a Web Application
- The model is deployed using **Streamlit**, a Python framework for web apps.
- The user uploads an image via the web app.
- The model predicts the emotion and displays the image alongside the result.
- The application supports real-time emotion recognition.
