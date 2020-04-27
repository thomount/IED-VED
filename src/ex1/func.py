import numpy
import random
import cv2

def cut(a, *args):
    type = args[0] if len(args) > 0 else 0
    if type == 0:
        return a
    b = numpy.zeros(a.shape, dtype=float)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            b[i][j] = a[i][j] if i <= 20 or j <= 20 else 0
    return b

def dct(a, *args):
    type = args[0] if len(args) > 0 else 0
    if type == 0:
        return cv2.dct(a)
    if type == 1:
        b = numpy.zeros(a.shape, dtype = float)
        for i in range(a.shape[0] >> 3):
            for j in range(a.shape[1] >> 3):
                bt = cv2.dct(a[(i<<3):(((i+1)<<3)), (j<<3):(((j+1)<<3))])
                for i1 in range(8):
                    for j1 in range(8):
                        b[8*i+i1, 8*j+j1] = bt[i1, j1]
                
        return b
    return a
def idct(a, *args):
    type = args[0] if len(args) > 0 else 0
    if type == 0:
        return cv2.idct(a)
    if type == 1:
        b = numpy.zeros(a.shape, dtype = float)
        for i in range(a.shape[0] >> 3):
            for j in range(a.shape[1] >> 3):
                bt = cv2.idct(a[(i<<3):(((i+1)<<3)), (j<<3):(((j+1)<<3))])
                for i1 in range(8):
                    for j1 in range(8):
                        b[8*i+i1, 8*j+j1] = bt[i1, j1]
        return b
    return a
