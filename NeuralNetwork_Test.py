#import NeuralNetwork_Training as train
from keras.models import load_model
import numpy as np
import cv2
import CC
from tkinter.filedialog import askopenfilenames
from constant import *
from keras import backend as K


# acquisisce l'immagine in input e la ritorna centrata in un'immagine che ha le dimensioni definite in constant.py
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


# prende in input una lista di immagini (le componenti connesse) e restituisce un numpy contenente le immagini con il
# formato necessario per svolgere il test
def createTestData(CCs):
    x_test = []                               # lista che conterra' tutte le immagini di taglia ridotta
    for cc in CCs:
        x_test.append(resizeImage(cc))

    x_test = np.asarray(x_test)                   # trasformo x_test in numpy array

    if K.image_data_format() == 'channels_first':
        x_test = x_test.reshape(x_test.shape[0], 1, IMG_ROWS, IMG_COLS)
    else:
        x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)

    x_test = x_test.astype('float32')
    x_test /= 255

    return x_test


# input: percorso della planimetria da testare
# output: lista delle componenti connesse della planimetria, classe di ognuna di essa
def test(components):
    CCs = []                   # lista delle componenti connesse (solo l'immagine, con dimensione dell'img intera)
    for ps in components:
        CCs.append(ps.image)

    x_test = createTestData(CCs)                  # formatta i dati su cui voglio svolgere il test

    model = load_model('model1.h5')                # carico il modello che ho creato e salvato con il training
    proba = model.predict_proba(x_test, batch_size=1)
    #print(proba)

    classes = model.predict_classes(x_test, batch_size=1)

    return proba
