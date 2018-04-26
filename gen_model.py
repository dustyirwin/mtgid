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
