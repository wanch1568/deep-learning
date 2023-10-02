import cv2
import numpy as np
from tensorflow import keras

# モデルとカテゴリをロード
model = keras.models.load_model('fer.h5')
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_emotion(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = np.reshape(face_img, [1, 48, 48, 1])
    
    prediction = model.predict(face_img)
    return emotion_classes[np.argmax(prediction)]

# 画像から顔を検出
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv2.imread('new_face.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# 検出された各顔の感情を予測
for (x, y, w, h) in faces:
    face = img[y:y+h, x:x+w]
    emotion = predict_emotion(face)
    
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

cv2.imshow('Emotion Recognition', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
