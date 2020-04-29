import numpy
import random
import cv2

def cut(a, *args):
    type = args[0] if len(args) > 0 else 0
    if type == 0:
        return a
    if type == 1:
        b = numpy.zeros(a.shape, dtype=float)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                b[i][j] = a[i][j] if i < (a.shape[0]>>1) and j < (a.shape[1]>>1) else 0
        return b


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
