import CC
import cv2
import numpy as np
import json
import os.path
from constant import *
from PIL import Image
from tkinter.filedialog import askopenfilenames
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


def resizeImage(img):
    x, y, w, h = cv2.boundingRect(img)
    l = max(w, h)
    l += int(l / 5)
    x -= int((l - w) / 2)
    y -= int((l - h) / 2)

    if x < 0 or y < 0:
        tmp = np.zeros((l, l))
        if x < 0 and y >= 0:
            #tmp[-x:, 0:] = img[0:l + x, y:y + l].copy
            tmp = img[y:y + l, :l]
        if x >= 0 and y < 0:
            #tmp[0:, -y:] = img[x:x + l, 0:l + y].copy()
            tmp = img[:l, x:x + l]
        if x < 0 and y < 0:
            #tmp[-x: -y] = img[0:l + x, 0:l + y].copy()
            tmp = img[:l, :l]
        img = tmp
    else:
        img = img[y:y + l, x:x + l].copy()
    img = cv2.resize(img, (IMG_ROWS, IMG_COLS))

    return img


def createTrainingSet(files):
    x_train = []                   # lista che conterra' tutte le componenti connesse, che costituiranno il training set
    y_train = []                   # y_train[i] dice se i-esima componente connessa e' parte di una scala (1) o no (0)

    for imagePath in files:
        # eseguo codice che restituisce tutte le componenti connesse
        num, lab = CC.connectedComponents(imagePath)
        images = CC.allComponents(num, lab)

        # apro etichettatura e creo immagine della scala che e' stata etichettata
        labeledImg = getLabeledImg(imagePath, images[0].image.shape)

        for posStep in images:
            img = posStep.image
            #Image.fromarray(img).save("C:/Users/manue/Desktop/img.bmp")

            # controllo se la componente che sto verificando e' parte della scala etichettata o meno
            unique, counts = np.unique(img, return_counts=True)
            dictInter = dict(zip(unique, counts))
            if 255 not in dictInter.keys():
                nPixel = 0
            else:
                nPixel = dictInter[255]

            intersection = cv2.bitwise_and(labeledImg, img)

            unique, counts = np.unique(intersection, return_counts=True)
            dictInter = dict(zip(unique, counts))
            if 255 not in dictInter.keys():
                nPixel2 = 0
            else:
                nPixel2 = dictInter[255]

            x_train.append(resizeImage(img))
            if nPixel2 > nPixel/2:
                y_train.append(1)                    # se componente connessa e' parte della scala
            else:
                y_train.append(0)                    # se NON e' parte della scala

    # se il numero di campioni negativi e' >> di quelli positivi, ne elimino alcuni
    pos, neg = 0, 0
    for el in y_train:
        if el == 0:
            neg += 1
        else:
            pos += 1

    if neg > pos*3:
        while len(y_train) > pos*5/2:
            index = np.random.randint(0, len(y_train), 1)
            index = index[0]
            if y_train[index] == 0:
                y_train.__delitem__(index)
                x_train.__delitem__(index)

    return np.asarray(x_train), y_train


def getLabeledImg(path, shape):
    # cerco file .json corrispondente
    labelPath = path.split('.')
    labelPath = labelPath[0] + ".json"

    labeledImg = np.zeros(shape, np.uint8)

    if os.path.exists(labelPath):
        with open(labelPath) as f:          # leggo file .json
            data = json.load(f)

        contours = []
        for o in data[0]["annotations"]:
            if o["class"] == "Stairs":         # prendo solo i dati relativi alle etichette di scale
                x = o["xn"].split(';')
                y = o["yn"].split(';')

                for i in range(len(x)):
                    contours.append([[int(float(x[i])), int(float(y[i]))]])

        contours = np.array(contours, dtype=np.int32)
        cv2.drawContours(labeledImg, [contours], -1, 255, cv2.FILLED)  # creo immagine della scala etichettata

    return labeledImg


def training():
    batch_size = 128
    num_classes = 2
    epochs = 12

    # choose the files and split them into train and test
    files = askopenfilenames()  # show an "Open" dialog box and return the path to the selected file
    for file in files:
        print(file)

    x, y = createTrainingSet(files)

    pos, neg = 0, 0
    for e in y:
        if e == 0:
            neg += 1
        else:
            pos += 1

    print("pos:", pos, "neg:", neg)

    x_train = np.zeros((1,IMG_ROWS,IMG_COLS))
    y_train = []
    x_test = np.zeros((1,IMG_ROWS,IMG_COLS))
    y_test = []

    indexes = np.arange(0,len(x))              # creo array con elementi casuali che indicano quali indici andranno usati per il validation (test)
    np.random.shuffle(indexes)
    indexes = indexes[0:int(len(x)/5)].copy()

    # divido le componenti che ho trovato in train (80%) e validation (20%)
    for i in range(len(x)):
        if i in indexes:
            x_test = np.append(x_test, x[i].reshape((1,IMG_ROWS,IMG_COLS)), axis=0)
            y_test.append(y[i])
        else:
            x_train = np.append(x_train, x[i].reshape((1,IMG_ROWS,IMG_COLS)), axis=0)
            y_train.append(y[i])

    x_train = np.delete(x_train, [0], axis=0)
    y_train = np.asarray(y_train)
    x_test = np.delete(x_test, [0], axis=0)
    y_test = np.asarray(y_test)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, IMG_ROWS, IMG_COLS)
        x_test = x_test.reshape(x_test.shape[0], 1, IMG_ROWS, IMG_COLS)
        input_shape = (1, IMG_ROWS, IMG_COLS)
    else:
        x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
        x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)
        input_shape = (IMG_ROWS, IMG_COLS, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('my_model.h5')


training()