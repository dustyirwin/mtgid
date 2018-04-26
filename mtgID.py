"""
Class for training a CNN on MtG cards for set prediction.
"""

import keras
import json
import requests
import numpy as np
from PIL import Image
from keras.preprocessing import image as kim


class mtgId:
    def __init__(self):
        global model
        with open('AllSets.json') as json_data:
            self.sets = json.load(json_data)
        model = keras.models.load_model("setm.h5")
        print("CNN loaded from h5 file.")
        self.setVec = [(s, self.sets[s]['releaseDate']) for s in self.sets.keys()]
        self.setVec.sort(key=lambda x: x[1])
        print(str(len(self.setVec)) + " MtG sets loaded from JSON file. ")

    def learn_set(self, setCode):
        global model
        self.xs, self.ys = self.get_set_data(setCode)
        model.fit(
            self.xs,
            self.ys,
            epochs=1,
            verbose=1,
            validation_split=0.1,
            batch_size=16)
        print("Training complete.")

    def predict_set(self, setCode, cardNum):
        global model
        if setCode in self.sets.keys():
            self.setHot = self.get_set_hot(setCode)
            self.multiverseId = str(self.sets[setCode]['cards'][cardNum]['multiverseid'])
            self.img = np.array([self.get_card_image(self.multiverseId)[0]])
            self.setPred = model.predict(self.img).tolist()[0]
            self.setPredIndex = self.setPred.index(max(self.setPred))
            self.setPredName = self.setVec[self.setPredIndex][0]
            return self.setPredName
        else:
            print('SetCode not found in sets.keys():', self.sets.keys())

    def get_card_image(self, multiverseId):
        self.imgURL = "http://gatherer.wizards.com/Handlers/Image.ashx?multiverseid=" + str(multiverseId) + "&type=card"
        self.imgRGB = Image.open(requests.get(self.imgURL, stream=True).raw)
        self.imgArray = kim.img_to_array(self.imgRGB)
        self.imgArray = self.imgArray[:310, :223, :3]
        self.imgArray = np.array(self.imgArray)
        return self.imgArray, self.imgRGB

    def get_set_hot(self, setCode):
        self.setHot = []
        for j in self.setVec:
            if j[0] == setCode:
                self.setHot.append(1)
            else:
                self.setHot.append(0)
        return self.setHot

    def get_set_data(self, setCode):
        self.xs = []
        self.ys = []
        for i in self.sets[setCode]['cards']:
            self.xs.append(self.get_card_image(i['multiverseid'])[0])
            self.ys.append(self.get_set_hot(setCode))
        print("Dataset for " + setCode + " created.")
        return np.array(self.xs), np.array(self.ys)

global model
mtgid = mtgId()

def grind_sets():
    while True:
        try:
            for setCode in mtgid.sets.keys():
                mtgid.learn_set(setCode)
                mtgid.predict_set(setCode, 1)
                model.save("setm.h5")
        except:
            continue

mtgid.predict_set("XLN", 5)
#mtgid.learn_set("XLN")
#model.save("setm.h5")

#mtgid.setPrediction
#score = mtgid.model.evaluate(x_test, y_test, batch_size=32)

"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='selu', input_shape=(310, 223, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='selu'))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='selu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='selu'))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='selu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='selu'))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='selu'))
model.add(Dropout(0.5))
model.add(Dense(219, activation='softmax'))

#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='ADAM', metrics=['accuracy'])
"""
