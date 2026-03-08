import cv2
import numpy as np
from tensorflow.keras.models import load_model


class Video:

    def __init__(self):

        self.video = cv2.VideoCapture(0)

        self.model = load_model("final_gender_model.h5")

        self.face_detector = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml"
        )

        self.labels = ["Male", "Female"]

    def __del__(self):
        self.video.release()

    def get_frame(self):

        success, frame = self.video.read()

        if not success:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face, (32, 32))
            face = face / 255.0
            face = face.reshape(1, 32, 32, 1)

            prediction = self.model.predict(face, verbose=0)

            label = int(prediction[0][0] > 0.5)

            gender = self.labels[label]

            color = (0,255,0) if label==1 else (255,0,0)

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

            cv2.putText(
                frame,
                gender,
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

        ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes()