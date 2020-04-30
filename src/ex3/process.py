import sys
import matplotlib.pyplot as plt
import numpy
plt.rcParams['font.sans-serif']=['SimHei']
filename1 = '../../output/ex3/res0_'+sys.argv[1]+'.txt'
filename2 = '../../output/ex3/res1_'+sys.argv[1]+'.txt'

f1 = open(filename1, 'r')
f2 = open(filename2, 'r')

#vec = [[], []]
m = [[], []]

for i in range(100):
    parts1 = f1.readline().split()
    parts2 = f2.readline().split()
    #print(parts1, parts2)
    m[0].append(int(parts1[-1]))
    m[1].append(int(parts2[-1]))

#plt.clf()
plt.plot(m[0], 'o',color='r', label='基于像素')
plt.plot(m[1], '.',color='b', label='基于dct')
plt.legend(loc=2)
plt.title('MSE变化图')
plt.xlabel('帧')
plt.ylabel('MSE')
plt.savefig('../../output/ex3/mse_'+sys.argv[1]+'.png')

print('基于像素 mean mse =\t', numpy.mean(m[0]))
print('基于dct  mean mse =\t', numpy.mean(m[1]))
