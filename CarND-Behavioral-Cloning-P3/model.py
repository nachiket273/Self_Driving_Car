from tensorflow import keras
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Flatten, Dense, Activation, Cropping2D, Dropout
import utils

def get_model(optimizer, loss='mse'):
    model = Sequential()

    model.add(Lambda(lambda x: x /127.5 - 1.0, input_shape = (utils.IMG_HT, utils.IMG_WIDTH, utils.IMG_CH)))

    model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="elu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="elu", strides=(1, 1)))
    model.add(Conv2D(64, (3, 3), activation="elu", strides=(1, 1)))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer)

    return model