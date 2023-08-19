import cv2
import numpy as np
import os
import sys
from numpy import expand_dims
import h5py
from keras.models import load_model, model_from_json
import mediapipe as mp
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import mediapipe as mp
import pickle


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

# load the model

model = load_model('facenet_keras.h5')
model.load_weights('facenet_keras_weights.h5')
cap = cv2.VideoCapture(0)


def find_cosine_similarity(source_representation, test_representation):
    a = np.dot(source_representation, test_representation)
    b = np.sqrt(source_representation.dot(source_representation))
    c = np.sqrt(test_representation.dot(test_representation))
    return 1 - (a / (b*c))
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]



def load_data(): 
    f = open('embeddings.csfd', 'wb')
    faces = []
    for i in os.listdir('smallerfaces'):       
        try:
            if i.split('.')[1] == 'jpg':
                print('img found')
                img = cv2.imread('smallerfaces' + os.path.sep + i)
                temp = (get_embedding(model, cv2.resize(img, (160, 160))), i.split('.')[0])
                faces.append(temp)
        except:
            pass 
    print(faces)
    pickle.dump(faces, f)
    f.close()           
    return 69420		