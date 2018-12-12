# import the necessary packages
import numpy as np
import math
import cv2
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import PossibleStep as ps
import NeuralNetwork as nt


def connectedComponents(path):
    image = Image.open(path)

    # applico trasformazioni morfologiche alla mia immagine per migliorare eventuali errori presenti
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
    #Image.fromarray(image).save("C:/Users/manue/Desktop/gray.bmp")
    _, image = cv2.threshold(image, 250, 255, cv2.THRESH_BINARY)
    #Image.fromarray(image).save("C:/Users/manue/Desktop/bin.bmp")
    # kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)             # definisco elemento strutturante
    kernel = np.ones((2,2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    # image = cv2.dilate(image, kernel, iterations=1)
    #Image.fromarray(image).save("C:/Users/manue/Desktop/erose.bmp")

    # converto l'immagine in scale di grigio e poi la binarizzo
    # gray_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
    # _, th = cv2.threshold(gray_image, 127, 255, cv2.THRESH_OTSU)

    # cerco le componenti connesse
    numCC, labels = cv2.connectedComponents(image)
    # print (numCC)

    # labels e' una matrice singola in cui sono etichettate le componenti connesse
    return numCC, labels


def createGraph(components):
    G = nx.Graph()
    node_color = {}
    labels = {}

    tot = np.zeros(components[0].image.shape)
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            seg1 = [(0,0), (1,1)]
            seg2 = [(0,0), (1,1)]
            # cerco i due lati confinanti delle figure
            for k in range(len(components[i].contour)):
                if intersect(components[i].centroid, components[j].centroid, components[i].contour[k][0], components[i].contour[(k + 1) % len(components[i].contour)][0]):
                    if distanceLinePoint(components[i].contour[k][0], components[i].contour[(k + 1) % len(components[i].contour)][0], components[j].centroid) < distanceLinePoint(seg1[0], seg1[1], components[j].centroid):
                        seg1 = [components[i].contour[k][0], components[i].contour[(k + 1) % len(components[i].contour)][0]]  # lato della figura rects[i]
            for k in range(len(components[j].contour)):
                if intersect(components[i].centroid, components[j].centroid, components[j].contour[k][0], components[j].contour[(k + 1) % len(components[j].contour)][0]):
                    if distanceLinePoint(components[j].contour[k][0], components[j].contour[(k + 1) % len(components[j].contour)][0], components[i].centroid) < distanceLinePoint(seg2[0], seg2[1], components[i].centroid):
                        seg2 = [components[j].contour[k][0], components[j].contour[(k + 1) % len(components[j].contour)][0]]  # lato della figura rects[j]

            # controllo le distanze tra i vertici per vedere se le componenti sono vicine o no
            tup1 = (distance(seg1[0], seg2[0]), distance(seg1[1], seg2[1]))
            tup2 = (distance(seg1[0], seg2[1]), distance(seg1[1], seg2[0]))

            if abs(tup1[0] - tup1[1]) < abs(tup2[0] - tup2[1]) and min(tup1[0], tup1[1]) <= 6:
                tup = tup1
            else:
                tup = tup2

            objectDistance = min(tup[0], tup[1])  # objectDistance contiene la distanza tra i due oggetti

            if objectDistance <= 6:
                # print (i, "e", j, "confinano")
                if components[i] not in G:
                    G.add_node(components[i], pos=(components[i].centroid[0], -components[i].centroid[1]))
                    labels[components[i]] = i
                    node_color[components[i]] = "red"
                if components[j] not in G:
                    G.add_node(components[j], pos=(components[j].centroid[0], -components[j].centroid[1]))
                    labels[components[j]] = j
                    node_color[components[j]] = "red"
                # print (i, j)
                edge = createEdge(components[i], seg1.copy(), components[j], seg2.copy(), tup)

                if edge != None:
                    G.add_edge(components[i], components[j], feature=edge)
                    if isRectangleLike(components[i].image):
                        node_color[components[i]] = "yellow"
                    else:
                        node_color[components[i]] = "orange"

                    if isRectangleLike(components[j].image):
                        node_color[components[j]] = "yellow"
                    else:
                        node_color[components[j]] = "orange"

                tot += components[i].image
                tot += components[j].image

    #cv2.drawContours(tot, [components[32].contour], -1, 127, 2)
    #cv2.drawContours(tot, [components[52].contour], -1, 127, 2)
    #print (cv2.contourArea(components[87].contour))

    tot = tot.astype(np.uint8)
    #Image.fromarray(tot).save("C:/Users/manue/Desktop/imgTot.bmp")

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_color=list(node_color.values()))
    nx.draw_networkx_labels(G, pos, labels)

    plt.show()

    # elimino le componenti connesse con meno di 5 elementi
    dictCC = {}
    cont = 0

    for sg in list(nx.connected_component_subgraphs(G, copy=False)):
        if len(sg.nodes()) <= 5:
            stairs = 0
            #for node in sg.nodes():                   # condizione aggiunta se uso versione con rete neurale
               # if node.probability > 0.5:
              #      stairs += 1
            if stairs < 3:
                for node in sg.nodes():
                    del node_color[node]
                    del labels[node]
                    G.remove_node(node)
                    tot[node.image != 0] = 0
            else:
                dictCC[cont] = np.zeros(tot.shape, np.uint8)
                for n in sg.nodes():
                    dictCC[cont] += n.image
                dictCC[cont] = dictCC[cont].astype(np.uint8)
                cont += 1
        # elimino anche le componenti connesse in cui la maggioranza degli archi ha intersezione!=1
        else:
            edges = sg.edges(data=True)
            perfectIntersection = 0
            for e in edges:
                #print (e[2]["feature"].intersection)
                if e[2]["feature"].intersection >= 0.9:
                    perfectIntersection += 1
            if perfectIntersection < len(edges)/4 + 1:
                for node in sg.nodes():
                    del node_color[node]
                    del labels[node]
                    G.remove_node(node)
                    tot[node.image != 0] = 0
            # creo dizionario in cui ogni voce e' una componene connessa
            else:
                dictCC[cont] = np.zeros(tot.shape, np.uint8)
                for n in sg.nodes():
                    dictCC[cont] += n.image
                dictCC[cont] = dictCC[cont].astype(np.uint8)
                cont += 1

    tot = tot.astype(np.uint8)
    #Image.fromarray(tot).save("C:/Users/manue/Desktop/imgTotRid.bmp")

    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_color=list(node_color.values()))
    nx.draw_networkx_labels(G, pos, labels)

    plt.show()

    return dictCC


def createEdge(nodeI, segI, nodeJ, segJ, tup):
    # calcolo angolo, centrato nel centro della figura, che tocca i due estremi del touching segment
    # angolo nodoI
    '''d = distanceLinePoint(segI[0], segI[1], nodeI.centroid)
    r0 = distance(nodeI.centroid, segI[0])
    r1 = distance(nodeI.centroid, segI[1])
    angI = math.acos(d / r0) + math.acos(d / r1)
    angI = angI / 3.14 * 180

    # angolo nodoJ
    d = distanceLinePoint(segJ[0], segJ[1], nodeJ.centroid)
    r0 = distance(nodeJ.centroid, segJ[0])
    r1 = distance(nodeJ.centroid, segJ[1])
    angJ = math.acos(d / r0) + math.acos(d / r1)
    angJ = angJ / 3.14 * 180

    # faccio il rapporto tra le aree
    areaI = cv2.contourArea(nodeI.contour)
    areaJ = cv2.contourArea(nodeJ.contour)
    ratioArea = min(areaI, areaJ) / max(areaI, areaJ)
    '''
    # calcolo in quanto intersecano i due rettangoli
    inter1 = intersectionPercentual(segI, segJ, tup)
    inter2 = intersectionPercentual(segJ, segI, tup)
    if inter1 == 0 and inter2 != 0:
        inter = inter2
    elif inter1 != 0 and inter2 == 0:
        inter = inter1
    else:
        inter = min(inter1, inter2)

    #print (inter)
    if inter <= 0.4:
        return None

    #return ps.Edge(nodeI, nodeJ, angI, angJ, inter)
    return ps.Edge(nodeI, nodeJ, 0, 0, inter)


def allComponents(num, labels):
    components = []
    area = labels.shape[0]*labels.shape[1]
    minArea = area/28000
    maxArea = area/80
    tot = np.zeros(labels.shape)


    for i in range(2, num):
        CC = np.copy(labels)
        CC[CC != i] = 0
        CC[CC != 0] = 255  # se mettevo CC[CC==i] non funzionava (?)

        CC = CC.astype(np.uint8)  # converto in uint8 perche alcune funzioni di opencv necessitano di caratteri uint8
        # Image.fromarray(CC).show()

        # stampo il lavoro svolto
        if i == num-1:
            print ("count: ", i, "/", num-1)

        # trovo i contorni della figura rappresentata dalla CC
        image, contours, hier = cv2.findContours(CC, 1, 2)

        cnt = contours[0]
        peri = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(CC)
        a = len(cnt)
        b = cv2.contourArea(cnt)
        # per ogni contorno con l'area all'interno di un certo range: trovo il centro e creo un oggetto PossibleStep che lo descrive
        if len(cnt) > 2 and cv2.contourArea(cnt) >= minArea and cv2.contourArea(cnt) <= maxArea and w <= labels.shape[1]/5 and h <= labels.shape[0]/5:
            tot += CC
            M = cv2.moments(cnt)
            centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            components.append(ps.PossibleStep([], CC.copy(), centroid, cnt))

    tot = tot.astype(np.uint8)
    #Image.fromarray(tot).save("C:/Users/manue/Desktop/allComponents.bmp")
    return components


def isRectangleLike(CC):
    y_rect, y_internal, centroid, contour = contoursDescriptor(CC.copy())
    if len(y_rect) == len(y_internal):
        dif = []
        for j in range(len(y_internal)):  # calcolo la distanza tra y_internal ed y_rect
            dif.append(y_rect[j] - y_internal[j])

        mean = sum(dif) / len(dif)
        if mean <= min(dif) + 0.2 and mean >= max(
                dif) - 0.2:  # se i due array sono abbastanza vicini significa che la CC rappresenta un rettangolo
            return True
    else:
        return False


def contoursDescriptor(CC):
    # find contours of CC and draw the bounding rectangle
    image, contours, hier = cv2.findContours(CC, 1, 2)
    if len(contours) > 1:
        maxContour = None
        areaContour = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > areaContour:
                areaContour = cv2.contourArea(cnt)
                maxContour = cnt
    else:
        maxContour = contours[0]

    x, y, w, h = cv2.boundingRect(maxContour)

    x -= 3  # aumento le dimensioni del rettangolo per essere sicuro che contenga la mia CC
    y -= 3
    w += 6
    h += 6

    cv2.rectangle(CC, (x, y), (x + w, y + h), 255, -1)  # disegno il bounding rectangle, come rettangolo pieno

    # find contours of bounding rectangle
    boundImage, boundContours, hier = cv2.findContours(CC.copy(), 1, 2)
    # print (boundContours[0])
    cnt = boundContours[0]

    # vertici del bounding rect
    vertex = [(cnt[0][0][0], cnt[0][0][1]), (cnt[1][0][0], cnt[1][0][1]), (cnt[2][0][0], cnt[2][0][1]),
              (cnt[3][0][0], cnt[3][0][1])]  # vertici del rettangolo

    vertical = distance(vertex[0], vertex[1])  # lunghezza (dimensione verticale) rettangolo
    horizontal = distance(vertex[0], vertex[3])  # larghezza (dimensione orizzontale) rettangolo

    centroid = [vertex[0][0] + (horizontal / 2), vertex[0][1] + (vertical / 2)]  # coordinate centro del rettangolo

    # calcolo l'angolo al quale finisce il lato verticale ed inizia quello orizzontale

    r = math.sqrt((vertical / 2) ** 2 + (horizontal / 2) ** 2)
    angMax = math.acos((horizontal / 2) / r)

    numSector = 45  # num di settori in cui voglio dividere il rettangolo
    gradeSector = 360 / numSector

    # dati per disegnare grafico su rettangolo esterno
    x = [gradeSector * i for i in range(numSector)]
    y_rect = []
    x_rect = []
    # dati per disegnare grafico su CC contenuta nel rettangolo
    x_internal = []
    y_internal = []

    for i in range(numSector):
        ang = ((3.14 * 2) / numSector) * i
        if ang <= angMax or ang > 2 * 3.14 - angMax:
            point = (centroid[0] + (horizontal / 2), centroid[1] + ((horizontal / 2) * math.tan(ang)))
        elif ang <= 3.14 - angMax:
            point = (centroid[0] + ((vertical / 2) * math.tan((3.14 / 2) - ang)), centroid[1] + (vertical / 2))
        elif ang <= 3.14 + angMax:
            point = (centroid[0] - (horizontal / 2), centroid[1] + ((horizontal / 2) * math.tan(3.14 - ang)))
        elif ang <= 2 * 3.14 - angMax:
            point = (centroid[0] + ((vertical / 2) * math.tan((3.14 / 2 * 3) - ang)), centroid[1] - (vertical / 2))

        l = distance(centroid, point)
        # print (gradeSector*i, ":     ", l)
        y_rect.append(l)
        x_rect.append(gradeSector*i)

        d = []
        for j in range(len(maxContour)):
            if intersect(centroid, point, maxContour[j][0], maxContour[(j + 1) % (len(maxContour))][0]):
                inters = line_intersection((centroid, point),
                                           (maxContour[j][0], maxContour[(j + 1) % (len(maxContour))][0]))
                x_internal.append(gradeSector * i)
                d.append(distance(centroid, inters))
        if d != []:
            y_internal.append(max(d))

    maxValue = max(y_rect)
    y_rect[:] = [x / maxValue for x in y_rect]
    y_internal[:] = [x / maxValue for x in y_internal]
    # plt.plot(x_rect, y_rect)
    # plt.plot(x_internal, y_internal)
    # plt.show()

    return y_rect, y_internal, centroid, maxContour


# restituisce la percentuale dell'intersezione di segI su segJ
def intersectionPercentual(segI, segJ, tup):
    a = tup[0]
    b = distanceLinePoint(segJ[0], segJ[1], segI[0])

    if a == b:
        rem1 = 0
    else:
        rem1 = math.sqrt(abs(a ** 2 - b ** 2))

    a = tup[1]
    b = distanceLinePoint(segJ[0], segJ[1], segI[1])

    if a == b:
        rem2 = 0
    else:
        rem2 = math.sqrt(abs(a ** 2 - b ** 2))

    if rem1 == 0 and rem2 == 0 and tup[0] > 6 and tup[1] > 6:
        return 0
    lung = distance(segJ[0], segJ[1])
    result = (lung - rem1 - rem2) / lung
    if result < 0:
        return 0
    return result


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(xdiff, ydiff)
    if div == 0:
        return False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def distance(a, b):
    # euclidian distance beetwen two points
    return math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))


def distanceLinePoint(p1, p2, p3):
    # distance beetwen p1p2 segment and point p3
    px = p2[0] - p1[0]
    py = p2[1] - p1[1]

    something = px * px + py * py

    u = ((p3[0] - p1[0]) * px + (p3[1] - p1[1]) * py) / float(something)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = p1[0] + u * px
    y = p1[1] + u * py

    dx = x - p3[0]
    dy = y - p3[1]

    dist = math.sqrt(dx * dx + dy * dy)

    return dist


def arrayDistance(A, B, j):
    tot = 0
    for i in range(len(A)):
        tot += (A[i] - B[(i + j) % len(A)]) ** 2
    tot = math.sqrt(tot)
    return tot


def main(imagePath):
    # imagePath = "/home/manuel/Scrivania/Tesi/thecape-sloth_v2-b65f503e9ab3/French-CVC/spain_T4_18.png"

    numCC, CCs = connectedComponents(imagePath)
    components = allComponents(numCC, CCs)
    #components = nt.main(components)
    return createGraph(components)
