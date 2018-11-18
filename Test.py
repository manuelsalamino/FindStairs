from PIL import Image
import json
import cv2
import numpy as np
import os.path
import CC
import xlsxwriter


def calculateJaccard(labeledImg, img):
    # calcolo l'interserzione delle due immagini e conto quanti pixel contiene
    intersection = cv2.bitwise_and(labeledImg, img)
    # Image.fromarray(intersection).show("intersection")
    unique, counts = np.unique(intersection, return_counts=True)
    dictInter = dict(zip(unique, counts))
    if 255 not in dictInter.keys():
        num = 0
    else:
        num = dictInter[255]

    # calcolo l'unione delle due immagini e conto quanti pixel contiene
    union = cv2.bitwise_or(labeledImg, img)
    #Image.fromarray(union).show("union")
    unique, counts = np.unique(union, return_counts=True)
    dictUnion = dict(zip(unique, counts))
    if 255 not in dictUnion.keys():
        if num == 0:
            den = 1
        else:
            den = 0
    else:
        den = dictUnion[255]

    return num/den


def calculateBestJaccard(labelPath, img):
    bestJaccard = 0      # salvo il miglior coeff di jaccard tra le varie etichettature presenti(potrebbero essere presenti piu scale etichettate in un file)
    labeledStairs = 0

    if os.path.exists(labelPath):
        with open(labelPath) as f:                             # leggo file .json
            data = json.load(f)

        for o in data[0]["annotations"]:
            labeledImg = np.zeros(img.shape, np.uint8)
            contours = []
            if o["class"] == "Stairs":                             # prendo solo i dati relativi alle scale etichettate
                x = o["xn"].split(';')
                y = o["yn"].split(';')

                labeledStairs += 1

                for i in range(len(x)):
                    contours.append([[int(float(x[i])), int(float(y[i]))]])

                contours = np.array(contours, dtype=np.int32)
                cv2.drawContours(labeledImg, [contours], -1, 255, cv2.FILLED)          # creo immagine della scala etichettata

                jaccard = calculateJaccard(labeledImg, img)          # calcolo IoU tra risultato del mio programma e scala etichettata
                if jaccard > bestJaccard:
                    bestJaccard = jaccard

    return bestJaccard, labeledStairs


foundTrue = 0           # numero di possibili scale con coeff di Jaccard > 0,8
found = 0             # numero di possibili scale trovate
correct = 0            # numero di scale sono state etichettate

# salvo i risultati in un file excel
workbook = xlsxwriter.Workbook('../result.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, 'num File')
worksheet.write(0, 1, 'coeff')
row = 1

for numImg in range(11, 12):
    # eseguo il codice per il riconoscimento delle scale
    #imagePath = "C:/Users/manue/Desktop/Manuel/scuola/Universita'/Tesi/French-CVC/spain_T4_"+str(numImg)+".png"
    imagePath = "C:/Users/manue/Desktop/plan/" + str(numImg) + ".png"
    images = CC.main(imagePath)              # creo lista contenente le immagini di tutte le possibili scale

    jaccard = {}

    for index, img in images.items():
        # converto il risultato in modo da poter effettuare operazioni sulle componenti trovate
        img = img.astype(np.uint8)
        img[img != 0] = 255
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)

        a = np.count_nonzero(img)

        # se esiste il file json con l'etichettatura della scala, creo immagine che rappresenta questa etichettatura,
        # altrimenti se non è presente il file json considero come se non ci fosse etichettatura e quindi creo un'immagine tutta nera
        lastImagePath = imagePath
        imagePath = imagePath.split('.')
        imagePath = imagePath[0] + ".json"

        jaccard[index], nCorrect = calculateBestJaccard(imagePath, img)
        if imagePath != lastImagePath:
            correct += nCorrect

    # controllo se le composizioni danno risultati migliori
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            bestJaccard, _ = calculateBestJaccard(imagePath, images[i]+images[j])
            if bestJaccard > jaccard[i] and bestJaccard > jaccard[j]:
                images[i] = images[i] + images[j]
                jaccard[i] = bestJaccard
                del images[j]
                del jaccard[j]
                break

    for jacc in list(jaccard.values()):
        found += 1
        if jacc > 0.8:
            foundTrue += 1
        print(numImg, jacc)

        worksheet.write(row, 0, numImg)
        worksheet.write(row, 1, jacc)
        row += 1

worksheet.write(2, 4, 'Jaccard')
worksheet.write(2, 5, 'cont')

jac = 0.7
for val in range(3, 8):
    worksheet.write(val, 4, jac)
    worksheet.write_formula(val, 5, '=CONTA.SE(B2:B'+str(row)+';">'+str(jac)+'")')
    jac += 0.05

val += 2

print ("Trovati giusti:", foundTrue)
worksheet.write(val, 4, 'Trovati giusti')
worksheet.write(val, 5, foundTrue)
val += 1

print ("Trovati:", found)
worksheet.write(val, 4, 'Trovati')
worksheet.write(val, 5, found)
val += 1

print ("Giusti:", correct)
worksheet.write(val, 4, 'Giusti')
worksheet.write(val, 5, correct)
val += 1

print ("Precision:", foundTrue/found)
worksheet.write(val, 4, 'Precision')
worksheet.write(val, 5, foundTrue/found)
val += 1

print ("Recall:", foundTrue/correct)
worksheet.write(val, 4, 'Recall')
worksheet.write(val, 5, foundTrue/correct)

workbook.close()