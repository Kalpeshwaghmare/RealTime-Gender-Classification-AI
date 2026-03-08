import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("final_gender_model.h5")

# Labels
labels = ["Male", "Female"]

# Colors for each gender
color_dict = {0: (255,0,0), 1: (0,255,0)}

# Face detector
faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Webcam
video = cv2.VideoCapture(0)

while True:

    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceDetector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30,30))

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (32, 32))
        face = face / 255.0
        face = face.reshape(1, 32, 32, 1)

        prediction = model.predict(face, verbose=0)

        # Convert prediction to label index
        label = int(prediction[0][0] > 0.5)

        gender = labels[label]

        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color_dict[label], 2)

        # Draw label background
        cv2.rectangle(frame, (x, y-40), (x+w, y), color_dict[label], -1)

        # Put gender text
        cv2.putText(frame, gender, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255,255,255), 2)

    cv2.imshow("Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()