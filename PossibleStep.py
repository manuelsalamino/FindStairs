class PossibleStep:
    def __init__(self, des, img, cen, cont):
        self.descriptor = des
        self.image = img
        self.centroid = cen
        self.contour = cont
        self.probability = None

    def setProb(self, p):
        self.probability = p

class Edge:
    def __init__(self, nodeI, nodeJ, angI, angJ, inter):
        self.nodeI = nodeI
        self.nodeJ = nodeJ
        self.angI = angI
        self.angJ = angJ
        self.intersection = inter
