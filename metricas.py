#librerias
import numpy as np

#Calculo de las metricas Accuracy, Precision, Recall y F1-Score
def calcular_metricas(pred, gt):

    pred = pred>0
    gt = gt>0

    TP= np.logical_and(pred, gt).sum()
    TN= np.logical_and(~pred, ~gt).sum()
    FP= np.logical_and(pred, ~gt).sum()
    FN= np.logical_and(~pred, gt).sum()

    accuracy= (TP+TN)/(TP+TN+FP+FN+1e-8)
    precision= TP/(TP+FP+1e-8)
    recall = TP / (TP+FN+1e-8)
    f1 = 2*(precision*recall) /(precision+recall+1e-8)

    #Devolucion de valores
    return accuracy, precision, recall, f1
