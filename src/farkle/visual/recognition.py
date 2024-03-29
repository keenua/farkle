from os import listdir, path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

import cv2
import numpy as np

TRAIN_DIR = 'train'
MODEL_NAME = 'rec.model'
SHAPE = (25, 25)

model = None

lower_gray = np.array([50, 0, 0], np.uint8)
upper_gray = np.array([255, 255, 75], np.uint8)


def vector(img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

    blur = cv2.bitwise_and(img, img, mask=mask_gray)
    blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV)[1]

    result = cv2.resize(thresh, SHAPE, interpolation=cv2.INTER_NEAREST) // 255
    result = result[..., np.newaxis].astype(bool)
    return result


def prepare_samples():
    X = np.empty((0, SHAPE[0], SHAPE[1], 1), dtype=np.bool)
    Y = np.empty((0), dtype=np.byte)

    for i in range(0, 7):

        name = 'bg' if i == 0 else str(i)
        dir = path.join(TRAIN_DIR, name)

        for file in listdir(dir):
            filepath = path.join(dir, file)

            img = cv2.imread(filepath)
            vec = vector(img)
            resized = vec[np.newaxis, ...]

            X = np.append(X, resized, axis=0)
            Y = np.append(Y, i)

    return train_test_split(X, Y, test_size=0.1)


def train():
    x_train, x_test, y_train, y_test = prepare_samples()

    print(f'Train samples: {len(x_train)}')

    input_shape = (SHAPE[0], SHAPE[1], 1)

    model = Sequential()
    model.add(Conv2D(20, kernel_size=(4, 4), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(7, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x=x_train, y=y_train, epochs=10)

    model.evaluate(x_test, y_test)
    model.save(MODEL_NAME, overwrite=True)


def recognize(img: np.ndarray) -> int:
    global model
    if not model:
        model = load_model(MODEL_NAME)

    vec = vector(img)
    vec = vec[np.newaxis, ...]
    return model.predict(vec).argmax()


if __name__ == '__main__':
    train()
