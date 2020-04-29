import numpy
import cv2
import func
import sys
import math
import time
import matplotlib.pyplot as plt

choice = int(sys.argv[1]) if len(sys.argv) > 0 else 0
configs = [(0, 0), (1, 0), (2, 0) , (1, 4), (1, 16), (1, 64), (1,'Q'), (1, 'QC'), (1, 'QN')]

# 0:  2d-dct 100%           0   0           (auto)
# 1:  2d-dct 8*8 100%       1   0           (solve)
# 2:  1d-dct r&c 50%        2   0           (solve) 

# 3: 2d-dct ex 1/4          TODO
# 4: 2d-dct ex 1/16         TODO
# 5: 2d-dct ex 1/64         TODO

# 6: 2d-dct 100% Quantization
# 7: 2d-dct 100% Quantization canon
# 8: 2d-dct 100% Quantization nikon

config = list(configs[choice])+sys.argv[2:]

def show(d1, n1, d2, n2):
    cv2.imshow(n1, d1)
    cv2.imshow(n2, d2)
    cv2.waitKey(0)

d = cv2.imread('../../data/lena.bmp', 0)        #直接读入灰度图

df_before = d.astype(numpy.float32)

startTime = time.time_ns()
df_dct = func.dct(df_before, config[0]) 
df_dct = func.cut(df_dct, config[1:])
#df_dct1 = func.dct(df_before, 0) 
#df_dct1 = func.cut(df_dct1, 1);
#show(df_dct1.astype(numpy.uint8), "2d-dct", (df_dct-df_dct1).astype(numpy.uint8), "1d-dct")
df_after = func.idct(df_dct, config[0]) 
useTime = time.time_ns()-startTime
#print(df_after)
#show(d, "before", df_after.astype(numpy.uint8), "after")
loss = df_before-df_after
#print(loss)

MSE = numpy.mean(loss ** 2)
PSNR = 10*math.log(255*255/MSE, 10)
print('MSE = ', MSE)
print('PSNR = ', PSNR)
print('Time = ', useTime / 1000000)

dloss = [[func.getloss(loss[i<<3:(i+1)<<3, j<<3:(j+1)<<3]) for j in range(d.shape[1]>>3)] for i in range(d.shape[0]>>3)]
f = open("../../output/ex1/loss_"+str(choice)+'.txt', 'w')
print(dloss, file=f)
f.close()
cv2.imwrite("../../output/ex1/"+str(choice)+".bmp", df_after.astype(numpy.uint8))