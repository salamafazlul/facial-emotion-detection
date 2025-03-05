import numpy as np
import os
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras import models


model = models.load_model(r'facial-emotion-detection\human_emotion_classification.keras')

emotions = [['angry'],['disgust'],['fear'],['happy'],['neutral'],['sad'],['surprise']]

st.header('Human Emotion Recognition')
image_path = st.text_input('Enter Image Path')

image = cv2.imread(image_path)[:,:,0]
image = cv2.resize(image, (48,48))
image = np.invest(np.array([image]))

output = np.argmax(model.predict(image))
outcome = emotions[output]
stn = 'Emotion in the Image is '+ str(outcome)
st.markdown(stn)

image_name = os.path.basename(image_path)
st.image('test_images/' + image_name, width=300)