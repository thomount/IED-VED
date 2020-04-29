# 任务清单

## ex1: DCT/IDCT

### 图片转灰度（5）over
  
### 三种方法（20）over

    1D-DCT (solve)
    2D-DCT（auto）
    2D-DCT 8*8（auto）

经过一通查找，尴尬得发现其实先按行再按列做1d-dct就是2d-dct，怕不是个憨憨；
至于是dct时低通滤波，两个维度分别dct后低通滤波和2d-dct完再低通滤波效果也是一毛一样的...

### 比较（25） over

    时间复杂度（理论、实践）
        用优秀的numpy操作优化了手动实现部分的运行时间
    损失分析
        全图 2d-dct:
            MSE =  3.207169e-10
            PSNR =  143.0695851883563
            Time =  3.015
        8*8 2d-dct:
            MSE =  1.33973281288557e-12
            PSNR =  166.86062166577403
            Time =  36.931
        行列 1d-dct:
            MSE =  15.600791935686066
            PSNR =  36.19933716066726
            Time =  26.9277

### 延伸（50）asking 

    1/4,1/16,1/64 DCT系数
        如何选择系数
        重复三种方法
    思路、设计、结果、分析
        思路：
            1、选取较大的系数保留，较小的系数抹0
            2、量化后较大的系数保留，较小的系数抹0


## ex2: 量化

### 基础（30）over

    按8*8分块                   over
    DCT - 量化 - IDCT           over
    比较PSNR（单个+平均）       record

### 延伸：量化矩阵*a（70）

    a与PSNR变化曲线（10） record
    解释（20）            text
    分析量化矩阵影响因素，求更好的量化矩阵（20）      text
    对比Cannon&Nikon量化矩阵（20）                  record

## ex3:视频处理

### 基础（50）

    16*16分块          over
    像素 块匹配        over   
    压缩 块匹配        over
    按帧标定MVs        record
    按帧标定MSE        record

### 延伸 （50）

    比较两种匹配方法
    部分DCT系数(压缩域块匹配)
    部分像素(像素域块匹配)
    其他特征/更好匹配
    思路、设计、结果、分析
