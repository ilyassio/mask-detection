import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from skimage.feature import hog
import os

DATASET_PATH = './datasets/dataset/'
CATEGORIES = ['negatives', 'positives']
IMG_SIZE = 128

def read_data():
    X = []
    y = []

    for category in CATEGORIES:
        path = os.path.join(DATASET_PATH, category)
        label = CATEGORIES.index(category)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            features = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4), visualize=False)
            X.append(features.tolist())
            y.append(label)
        
    return np.array(X), np.array(y)

X, y = read_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation ="relu"),
    tf.keras.layers.Dense(128, activation ="relu"),
    tf.keras.layers.Dense(128, activation ="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer="adam", loss="BinaryCrossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=32, epochs=50, validation_split=0.2)

print('\nTest:', model.evaluate(X_test, y_test))

tf.keras.models.save_model(model, "model_tf")


