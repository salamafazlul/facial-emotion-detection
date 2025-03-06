# Human Emotion Recognition from Images using CNN

## Project Summary
This project focuses on building a human emotion recognition model using Convolutional Neural Networks (CNN) to classify emotions from images. The model takes an image as input and predicts the emotion depicted in it. The project follows a structured approach that includes data preprocessing, model training, validation, and deployment as a web application.

## Project Overview
The human emotion recognition model classifies emotions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. The model is trained using a large dataset of labeled images and then deployed as a web application using Streamlit, allowing users to upload an image and receive emotion predictions in real-time.

## Features & Functionality
- The model takes an image path as input.
- Predicts and displays the detected emotion.  
- Supports real-time emotion recognition.  
- Utilizes a dataset with seven emotion categories.  
- Tested for accuracy using a **[sample dataset](https://github.com/salamafazlul/facial-emotion-detection/tree/main/test_images)**.

## Technologies Used  
- **Programming:** Python  
- **Libraries:** TensorFlow, NumPy, OpenCV, Matplotlib  
- **Framework:** Streamlit (for web deployment)  
- **Model Training:** Keras Sequential API  
- **Data Handling:** Pandas  

## Data Preprocessing  
- The dataset consists of **28,000 training images** and **7,000 validation images**.  
- Each image is labeled based on its emotion.  
- Images are converted into arrays for model training.

## Model Architecture
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

## Model Training
- The model is trained using 28,000 training images and validated on 7,000 validation images.
- **Batch size:** 128
- **Epochs:** 50 (each epoch takes about 6 minutes to run)
- The trained model is saved to avoid retraining every time.

## Model Deployment as a Web Application
- The model is deployed using **Streamlit**, a Python framework for web apps.
- The user uploads an image via the web app.
- The model predicts the emotion and displays the image alongside the result.
- The application supports real-time emotion recognition.

## Results  
- **Angry → Correct**  
- **Happy → Correct**  
- **Neutral → Correct**  
- **Sad → Incorrectly predicted as Fear** *(due to similar facial expressions)*  
- **Overall Accuracy:** ~75% on sample images  

## App Screenshot
![App Screenshot](https://github.com/salamafazlul/facial-emotion-detection/blob/main/cover.png)

