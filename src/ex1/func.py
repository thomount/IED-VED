import numpy
import random
import cv2
import math

Q = [[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]]
QC = [[1,1,1,2,3,6,8,10],[1,1,2,3,4,8,9,8],[2,2,2,3,6,8,10,8],[2,2,3,4,7,12,11,9],[3,3,8,11,10,16,15,11],[3,5,8,10,12,15,16,13],[7,10,11,12,15,17,17,14],[14,13,13,15,15,14,14,14]]
QN = [[2,1,1,2,3,5,6,7],[1,1,2,2,3,7,7,7],[2,2,2,3,5,7,8,7],[2,2,3,3,6,10,10,7],[2,3,4,7,8,13,12,9],[3,4,7,8,10,12,14,11],[6,8,9,10,12,15,14,12],[9,11,11,12,13,12,12,12]]
def cut(a, *args):
    type = args[0] if len(args) > 0 else 0
    if type[0] == 0:
        return a
    if type[0] == 'Q':
        d = eval(type[1])
        for i in range(a.shape[0] >> 3):
            for j in range(a.shape[1] >> 3):
                a[i<<3:(i+1)<<3, j<<3:(j+1)<<3] = (a[(i<<3):(((i+1)<<3)), (j<<3):(((j+1)<<3))] / Q*(1/d)).astype(numpy.int)*Q*d
        return a
        
    if type[0] == 'QC':
        d = eval(type[1])
        for i in range(a.shape[0] >> 3):
            for j in range(a.shape[1] >> 3):
                a[i<<3:(i+1)<<3, j<<3:(j+1)<<3] = (a[(i<<3):(((i+1)<<3)), (j<<3):(((j+1)<<3))] / QC*(1/d)).astype(numpy.int)*QC*d
        return a
    if type[0] == 'QN':
        d = eval(type[1])
        for i in range(a.shape[0] >> 3):
            for j in range(a.shape[1] >> 3):
                a[i<<3:(i+1)<<3, j<<3:(j+1)<<3] = (a[(i<<3):(((i+1)<<3)), (j<<3):(((j+1)<<3))] / QN*(1/d)).astype(numpy.int)*QN*d
        return a




def dct(a, *args):
    type = args[0] if len(args) > 0 else 0
    if type == 0:
        return cv2.dct(a)
    if type == 1:
        b = numpy.zeros(a.shape, dtype = float)
        for i in range(a.shape[0] >> 3):
            for j in range(a.shape[1] >> 3):
                b[i<<3:(i+1)<<3, j<<3:(j+1)<<3] = cv2.dct(a[(i<<3):(((i+1)<<3)), (j<<3):(((j+1)<<3))])

        return b
    
    if type == 2:
        b = numpy.zeros(a.shape, dtype = float)
        c = numpy.zeros(a.shape, dtype = float)
        for i in range(a.shape[0]):
            b[i, :(a.shape[1]>>1)] = cv2.dct(a[i, :]).T[0, :(a.shape[1]>>1)]

        for i in range(a.shape[1] >> 1):
            c[:(a.shape[0]>>1), i] = cv2.dct(b[:, i].T)[:(a.shape[0]>>1)].T
        return c

    return a
def idct(a, *args):
    type = args[0] if len(args) > 0 else 0
    if type == 0:
        return cv2.idct(a)
    if type == 1:
        b = numpy.zeros(a.shape, dtype = float)
        for i in range(a.shape[0] >> 3):
            for j in range(a.shape[1] >> 3):
                b[i<<3:(i+1)<<3, j<<3:(j+1)<<3] = cv2.idct(a[(i<<3):(((i+1)<<3)), (j<<3):(((j+1)<<3))])

        return b
    if type == 2:
        b = numpy.zeros(a.shape, dtype = float)
        c = numpy.zeros(a.shape, dtype = float)
        for i in range(a.shape[0]):
            b[i, :]= cv2.idct(a[i, :])[:, 0]
        for i in range(a.shape[1]):
            c[:, i] = cv2.idct(b[:, i].T).T
        return c
    return a

def getloss(a):
    MSE = numpy.mean(a ** 2)
    PSNR = 10*math.log(255*255/MSE, 10)
    return (MSE, PSNR)