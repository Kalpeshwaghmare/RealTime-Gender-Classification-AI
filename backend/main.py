from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

model = load_model("../models/final_gender_model.h5")

labels = ["Male","Female"]

@app.get("/")
def home():
    return {"message":"AI Gender Detection API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    npimg = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image,(32,32))
    image = image/255.0
    image = image.reshape(1,32,32,1)

    prediction = model.predict(image)

    gender = labels[int(prediction[0][0] > 0.5)]

    return {"prediction": gender}