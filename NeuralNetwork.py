import NeuralNetwork_Test as test
from tkinter.filedialog import askopenfilenames
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
from constant import *
from keras.models import load_model
from keras.utils import plot_model


def main(possibleSteps):
    probs = test.test(possibleSteps)       # il test restituisce una lista delle componenti connesse e una lista con le rispettive classi

    tot = np.zeros(possibleSteps[0].image.shape)
    text = np.zeros(possibleSteps[0].image.shape)
    components = []

    for i in range(len(probs)):
        if probs[i][1] > 0.3:
            possibleSteps[i].setProb(probs[i][1])
            components.append(possibleSteps[i])
            tot += possibleSteps[i].image
            text += possibleSteps[i].image

            y, x = np.transpose(np.nonzero(possibleSteps[i].image))[0]

            text = Image.fromarray(text)
            draw = ImageDraw.Draw(text)
            draw.text((x,y), str(probs[i][1])[:4], 127)
            text = np.array(text, np.uint8)
    #Image.fromarray(text).save("C:/Users/manue/Desktop/imgText.bmp")
    tot = tot.astype(np.uint8)
    #Image.fromarray(tot).save("C:/Users/manue/Desktop/imgNN.bmp")

    return components
