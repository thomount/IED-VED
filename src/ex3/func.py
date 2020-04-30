import cv2
import numpy

def toGrey(a):
    return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)

def mark(b, pt, size):
    a = b.copy()
    a[pt[0], pt[1]:pt[1]+size] = [0, 0, 255]
    a[pt[0]+size, pt[1]:pt[1]+size] = [0, 0, 255]
    a[pt[0]:pt[0]+size, pt[1]] = [0, 0, 255]
    a[pt[0]:pt[0]+size, pt[1]+size] = [0, 0, 255]
    #cv2.imshow('pt=('+str(pt[0])+','+str(pt[1])+')', a)
    #cv2.waitKey(0)
    return a

def cut(a, pt, size, step):
    return a[pt[0]:pt[0]+size:step, pt[1]:pt[1]+size:step]

def mse(a):
    return numpy.sum(a**2)