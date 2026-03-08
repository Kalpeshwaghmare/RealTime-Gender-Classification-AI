import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

data = []
labels = []

dataset_path = "dataset"

for gender in ["male", "female"]:
    path = os.path.join(dataset_path, gender)

    label = 0 if gender == "male" else 1

    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (32,32))

        data.append(image)
        labels.append(label)

data = np.array(data) / 255.0
data = data.reshape(-1,32,32,1)

labels = np.array(labels)

X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.2)

model = Sequential()

model.add(Conv2D(32,(3,3),activation="relu",input_shape=(32,32,1)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile(
optimizer="adam",
loss="binary_crossentropy",
metrics=["accuracy"]
)

model.fit(X_train,y_train,epochs=10,validation_data=(X_test,y_test))

model.save("model.h5")

print("Model created successfully!")