# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 12:39:55 2019

@author: Shireen
"""

import cv2
import tensorflow as tf

CATEGORIES = ["Dog", "Cat"]

NAME = input("Give file Path: ")

def prepare(filepath):
    IMG_SIZE = 70  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("64x3-CNN.model")

prediction = model.predict([prepare(NAME)])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])