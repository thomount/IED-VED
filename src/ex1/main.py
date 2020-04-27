import numpy
import cv2
import func

def show(d1, n1, d2, n2):
    cv2.imshow(n1, d1)
    cv2.imshow(n2, d2)
    cv2.waitKey(0)

d = cv2.imread('../../data/lena.bmp', 0)        #直接读入灰度图

df_before = d.astype(numpy.float32)
df_dct = func.dct(df_before, 0) 

df_dct = func.cut(df_dct, 1)

df_after = func.idct(df_dct, 0) 

#print(df_after)
show(d, "before", df_after.astype(numpy.uint8), "after")