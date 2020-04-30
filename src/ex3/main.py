import cv2
import numpy
import func
import sys
k = 1 if len(sys.argv) == 1 else int(sys.argv[1])
Flag = 0 if len(sys.argv) <= 2 else int(sys.argv[2])
src = cv2.VideoCapture('../../data/cars.avi')
fourcc, fps, framesize, iscolor = int(src.get(cv2.CAP_PROP_FOURCC)), src.get(cv2.CAP_PROP_FPS), (int(src.get(cv2.CAP_PROP_FRAME_WIDTH)), int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))), True
print(fourcc, fps, framesize, iscolor)
dst = cv2.VideoWriter('../../output/ex3/bus'+str(Flag)+'_'+str(k)+'.avi',fourcc, fps, framesize, iscolor)
src.set(cv2.CAP_PROP_POS_FRAMES, 20)
#print(src.get(cv2.CAP_PROP_POS_FRAMES))

ret, img = src.read()
rpt = [115, 305]
size, W = 16, 30
func.mark(img, rpt, size)

#print(numpy.array(img).shape)


if Flag == 1:
    mod = func.cut(func.toGrey(img), rpt, size, 1).astype(numpy.int32)
    dmod = cv2.dct(mod.astype(numpy.float32))
    dmod = dmod[:size//k, :size//k]
else:
    dmod = func.cut(func.toGrey(img), rpt, size, k).astype(numpy.int32)
    mod = func.cut(func.toGrey(img), rpt, size, 1).astype(numpy.int32)
mvs = []
mses = []
f = open('../../output/ex3/res'+str(Flag)+'_'+str(k)+'.txt', 'w')
for T in range(100):
    print(T)
    ret, nimg = src.read()
    gimg = func.toGrey(nimg).astype(numpy.int32)
    mse, res, mret = 1e30, None, None
    for i in range(max(0, rpt[0]-W), min(rpt[0]+W, gimg.shape[0]-size)):
        for j in range(max(0, rpt[1]-W), min(rpt[1]+W, gimg.shape[1]-size)):
            
            if Flag == 1:
                tar = func.cut(gimg, [i, j], size, 1)
                dtar = cv2.dct(tar.astype(numpy.float32))[:size//k, :size//k]
            else:
                dtar = func.cut(gimg, [i, j], size, k)
                tar = func.cut(gimg, [i, j], size, 1)
            tmp = func.mse(dmod-dtar)
            if tmp < mse:
                mse = tmp
                res = [i, j]
                mret = tar
    mvs.append([res[0]-rpt[0], res[1]-rpt[1]])
    rpt = res
    mses.append(func.mse(mod-mret))
    ret = func.mark(nimg, res, size)
    #print(ret.shape)
    dst.write(ret)
    #cv2.imwrite('../../output/ex3/'+str(T)+'.bmp', ret)
    print(T, '\t', mvs[-1], '\t', mses[-1], file = f)
    #print(mod-mret, file = f)
    #print('', file = f)

#print(mvs)
#print(mses)
f.close()
src.release()
dst.release()
    

