import cv2
import tensorflow as tf
from skimage.feature import hog

MODEL = './model_tf'
IMG_SIZE = 128

model = tf.keras.models.load_model(MODEL)

labels_dict={0: 'Without mask', 1: 'With mask'}
color_dict={0: (0, 0, 255), 1: (0, 255, 0)}

camera = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

while True:
    check, frame = camera.read()
    frame_copy = frame.copy()
    frame_copy = cv2.flip(frame_copy, 1)
    img = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(img, 1.1, 5)

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_AREA)
        features = hog(face_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4), visualize=False)

        pred = model.predict(features.reshape(1,-1))
        label = 1 if pred[0][0] >= 0.6 else 0

        cv2.rectangle(frame_copy, (x,y), (x+w,y+h), color_dict[label], 2)
        cv2.rectangle(frame_copy, (x,y-40), (x+w,y), color_dict[label], -1)
        cv2.putText(frame_copy, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
    cv2.imshow('MASK DETECTOR', frame_copy)
    key = cv2.waitKey(10)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()