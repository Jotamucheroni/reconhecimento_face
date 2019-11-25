import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np
from PIL import Image
import matplotlib as pl

detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
reconhecimentoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")
indices = cPickle.load(open("recursos/indices_yale.pickle", 'rb'))
descritoresFaciais = np.load("recursos/descritores_yale.npy")
totalFaces = 0
totalAcertos = 0
VP = 0
VN = 0
FP = 0
FN = 0
limiar = 0.5
for arquivo in glob.glob(os.path.join("yalefaces/teste", "*.gif")):
    imagemFace = Image.open(arquivo).convert('RGB')
    imagem = np.array(imagemFace, 'uint8')
    idatual = int(os.path.split(arquivo)[1].split(".")[0].replace("subject", ""))
    totalFaces += 1
    facesDetectadas = detectorFace(imagem, 2)
    for face in facesDetectadas:
        e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
        pontosFaciais = detectorPontos(imagem, face)
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)
        listaDescritorFacial = [fd for fd in descritorFacial]
        npArrayDescritorFacial = np.array(listaDescritorFacial, dtype=np.float64)
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]
        #cont = 0
        distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)
        for i in range(len(distancias)):
            if distancias[i] <= limiar:
                distancias[i] = int(1)
                #cont += 1
            else:
                distancias[i] = int(0)
        #distancias = int(distancias)
        print("Distâncias: {}".format(distancias))
        minimo = np.argmin(distancias)
        distanciaMinima = distancias[minimo]

        nome = os.path.split(indices[minimo])[1].split(".")[0]
        #print(facesDetectadas)

        cv2.rectangle(imagem, (e, t), (d, b), (0, 0, 255), 2)
        texto = "{} {:.4f}".format(nome, distanciaMinima)
        cv2.putText(imagem, texto, (d, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255))

    cv2.imshow("Teste", imagem)
    cv2.destroyAllWindows()
    #print("Contador = ",cont)
    indice = {}
    idx = 0
    detectorFace = dlib.get_frontal_face_detector()
    indiceImagem = 0
    for arquivo in glob.glob(os.path.join("yalefaces/treinamento", "*.gif")):
        imagemFace = Image.open(arquivo).convert('RGB')
        imagem = np.array(imagemFace, 'uint8')
        facesDetectadasBase = detectorFace(imagem, 1)
        numeroFacesDetectadasBase = len(facesDetectadasBase)
        idcomparado = int(os.path.split(arquivo)[1].split(".")[0].replace("subject", ""))
        if idatual == idcomparado:
            if distancias[indiceImagem] == 1:
                VP += 1
            else:
                FN += 1
        else:
            if distancias[indiceImagem] == 0:
                VN += 1
            else:
                FP += 1



        for face in facesDetectadas:
            indice[idx] = arquivo
            idx += 1
            cv2.imshow("Treinamento", imagem)
            #cv2.waitKey(0)
            cv2.destroyAllWindows()
        indiceImagem += 1

print("VP = %d" % VP)
print("VN = %d" % VN)
print("FP = %d" % FP)
print("FN = %d" % FN)
acuracia = ((VP + VN) / (VP + VN + FP + FN)) * 100
taxa_FP = ((FP) / (FP + VN)) * 100
taxa_FN = ((FN) / (FN + VP)) * 100
precisao = ((VP) / (VP + FP)) * 100
especificidade = ((VN) / (VN + FP)) *100
revocacao = ((VP) / (VP + FN)) * 100
F1 = ((2*(precisao * revocacao)) / (precisao + revocacao))
print("Acurácia: {}".format(acuracia))
print("Taxa de FP: {}".format(taxa_FP))
print("Taxa de FN: {}".format(taxa_FN))
print("Precisao: {}".format(precisao))
print("Revocacao: {}".format(revocacao))
print("Especificidade: {}".format(especificidade))
print("F1: {}".format(F1))
print("Diferença da Taxa de FP - Taxa de FN: {}" .format(FP - FN))