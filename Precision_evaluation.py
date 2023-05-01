from __future__ import division, print_function
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pylab as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve
from sklearn import preprocessing
import gdal
import numpy as np
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn import datasets
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import gdal
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import StratifiedKFold
os.environ['PROJ_LIB'] = r'/home/cv/anaconda3/envs/test/share/proj'
cuda = torch.cuda.is_available()
np.random.seed(1)
def read(path):
    in_ds = gdal.Open(path)              # 读取要切的原图
    print("open tif file succeed")
    width = in_ds.RasterXSize                         # 获取数据宽度
    height = in_ds.RasterYSize                        # 获取数据高度
    outbandsize = in_ds.RasterCount                   # 获取数据波段数
    im_geotrans = in_ds.GetGeoTransform()             # 获取仿射矩阵信息
    im_proj = in_ds.GetProjection()                   # 获取投影信息
    datatype = in_ds.GetRasterBand(1).DataType
    x = in_ds.ReadAsArray(0,0,width,height)     #获取数据.astype(np.float)
    x = x.reshape(-1, 1)

    # max_index = np.unravel_index(np.argmax(x, axis=None), x.shape)#获取数据最大值及索引位置
    # max_value = x[max_index]
    # print(max_index, max_value)
    del in_ds
    return x

def AUCC(alldata_sort,tnsdata_sort,side,flag):
    re = np.searchsorted(alldata_sort, tnsdata_sort, side=side)  # rightleft
    # print(re)
    re2 = re / 2758643
    # print(re2)
    # print("--------------------------")

    nnum = 0
    numhe = []
    print("len(re2)",len(re2))
    for iii in range(1, 101):
        fflag = 100 - iii
        re3 = np.where(np.logical_and(re2 <= (100 / 100), re2 >= (fflag / 100)))
        nnum = nnum + len(re3[0])
        jilu = str(fflag + 1) + ":" + str(nnum)
        numhe.append(jilu)
    print(nnum)
    print(flag,nnum / (len(re2) * 100))
    print(numhe)
    return nnum /(len(re2) * 100)

def f1(p,r):
    return 2*p*r/(p+r)


X2_pns50n=[]
join_pns_r,join_pns_l,join_tns_r,join_tns_l,join_test_ac,join_pns_av,join_tns_av,join_pns_f,join_tns_f,join_alldata50_a=[],[],[],[],[],[],[],[],[],[]
join_all80,join_all60,join_all40,join_all20,join_all0=[],[],[],[],[]

X2_pns_r,X2_pns_l,X2_tns_r,X2_tns_l,X2_test_ac,X2_pns_av,X2_tns_av,X2_pns_f,X2_tns_f, X2_alldatatest50_a=[],[],[],[],[],[],[],[],[],[]
X2_all80,X2_all60,X2_all40,X2_all20,X2_all0=[],[],[],[],[]

name =[]
#ospath is the fold of output image of contrastive network
ospath = r'jufu25'
for i in os.listdir(ospath):
    if "tif" in i:
        print("______________________________________________________________")
        print(i)
        name.append(i)
        path2 = ospath +'/' + i
        X2 = read(path2)
        # path is the fold of output image of PU-baggingDT
        path = r'bag250/result691000real1501.tif'
        # print(path)
        X1 = read(path)
        BILI=1
        X1=np.around(X1/BILI, decimals=0)
        X2=np.around(X2/BILI, decimals=0)

        join=(np.around(X1/1, decimals=0)+np.around(X2, decimals=0))/2

        in_ds = gdal.Open(r'../newlabel2/label54.tif')  # 读取要切的原图
        # print("open tif file succeed")
        width = in_ds.RasterXSize  # 获取数据宽度
        height = in_ds.RasterYSize
        # get the tif parameters
        im_geotrans = in_ds.GetGeoTransform()  # 获取仿射矩阵信息
        im_proj = in_ds.GetProjection()
        s = in_ds.ReadAsArray(0, 0, width, height)  # 获取数据.astype(np.float)
        s2 = s.flatten()
        del in_ds

        allnum =len(s2)
        anll =  np.arange(allnum)
        pns = anll[s2 == 2]
        tns = anll[s2 == 1]

        allnum = anll[s2 < 3]

        X1_alldata = X1[allnum].reshape(1,-1)[0]
        X1_pns_data = X1[pns].reshape(1, -1)[0]
        X1_tns_data = X1[tns].reshape(1, -1)[0]
        #概率大于0.5的个数
        # join_alldata50 = len(np.where(join_alldata>=(5000/BILI))[0])

        X1_alldata_sort = np.sort(np.around(X1_alldata/1, decimals=0),axis = 0)
        X1_pns_data_sort = np.sort(np.around(X1_pns_data/1, decimals=0),axis = 0)
        X1_tns_data_sort = np.sort(np.around(X1_tns_data / 1, decimals=0), axis=0)
        X1_pns_right = AUCC(X1_alldata_sort, X1_pns_data_sort, side="right", flag="X1pnsright:")
        X1_pns_left = AUCC(X1_alldata_sort, X1_pns_data_sort, side="left", flag="X1pnsleft:")
        X1_tns_right = AUCC(X1_alldata_sort, X1_tns_data_sort, side="right", flag="X1tnsright:")
        X1_tns_left = AUCC(X1_alldata_sort, X1_tns_data_sort, side="left", flag="X1tnsleft:")



        join_alldata = join[allnum].reshape(1,-1)[0]
        join_pns_data = join[pns].reshape(1, -1)[0]
        join_tns_data = join[tns].reshape(1, -1)[0]
        #概率大于0.5的个数
        # join_alldata50 = len(np.where(join_alldata>=(5000/BILI))[0])

        join_alldata_sort = np.sort(np.around(join_alldata/1, decimals=0),axis = 0)
        join_pns_data_sort = np.sort(np.around(join_pns_data/1, decimals=0),axis = 0)
        join_tns_data_sort = np.sort(np.around(join_tns_data / 1, decimals=0), axis=0)
        join_pns_right = AUCC(join_alldata_sort, join_pns_data_sort, side="right", flag="joinpnsright:")
        join_pns_left = AUCC(join_alldata_sort, join_pns_data_sort, side="left", flag="joinpnsleft:")
        join_tns_right = AUCC(join_alldata_sort, join_tns_data_sort, side="right", flag="jointnsright:")
        join_tns_left = AUCC(join_alldata_sort, join_tns_data_sort, side="left", flag="jointnsleft:")
        #测试概率大于0.5的个数
        join_test50 = len(np.where(join_tns_data >= (5000))[0])
        join_test50_ac = join_test50 / 351

        join_alldata50 = len(np.where(join_alldata>=(5000))[0])
        join_alldata50_ac = join_alldata50/2758643*100
        join_alldata50_a.append(join_alldata50_ac)

        join_alldata80 = len(np.where(join_alldata >= (8000))[0])
        join_alldata60 = len(np.where(join_alldata >= (6000))[0]) - join_alldata80
        join_alldata40 = len(np.where(join_alldata >= (4000))[0]) - join_alldata80 - join_alldata60
        join_alldata20 = len(np.where(join_alldata >= (2000))[0]) - join_alldata80 - join_alldata60 - join_alldata40
        join_alldata0 = len(np.where(join_alldata >= 0)[0]) - join_alldata80 - join_alldata60 - join_alldata40 - join_alldata20


        join_pns_data80 = len(np.where(join_pns_data>=(8000))[0])
        join_pns_data60 = len(np.where(join_pns_data >= (6000))[0])-join_pns_data80
        join_pns_data40 = len(np.where(join_pns_data >= (4000))[0])-join_pns_data80-join_pns_data60
        join_pns_data20 = len(np.where(join_pns_data >= (2000))[0])-join_pns_data80-join_pns_data60-join_pns_data40
        join_pns_data0 = len(np.where(join_pns_data>=0)[0])-join_pns_data80-join_pns_data60-join_pns_data40-join_pns_data20

        join_tns_data80 = len(np.where(join_tns_data>=(8000))[0])
        join_tns_data60 = len(np.where(join_tns_data >= (6000))[0])-join_tns_data80
        join_tns_data40 = len(np.where(join_tns_data >= (4000))[0])-join_tns_data80-join_tns_data60
        join_tns_data20 = len(np.where(join_tns_data >= (2000))[0])-join_tns_data80-join_tns_data60-join_tns_data40
        join_tns_data0 = len(np.where(join_tns_data>=0)[0])-join_tns_data80-join_tns_data60-join_tns_data40-join_tns_data20


        # join_all80.append([join_alldata80/2758643,(join_pns_data80+join_tns_data80)/(351+526)])#join_pns_data80+join_pns_data60+join_pns_data40+join_pns_data20+join_pns_data0+
        # join_all60.append([join_alldata60/2758643,(join_pns_data60+join_tns_data60)/(351+526)])
        # join_all40.append([join_alldata40/2758643,(join_pns_data40+join_tns_data40)/(351+526)])
        # join_all20.append([join_alldata20/2758643,(join_pns_data20+join_tns_data20)/(351+526)])
        # join_all0.append([join_alldata0/2758643,(join_pns_data0+join_tns_data0)/(351+526)])

        join_all80.append([join_alldata80/2758643,(join_tns_data80)/(351)])#join_pns_data80+join_pns_data60+join_pns_data40+join_pns_data20+join_pns_data0+
        join_all60.append([join_alldata60/2758643,(join_tns_data60)/(351)])
        join_all40.append([join_alldata40/2758643,(join_tns_data40)/(351)])
        join_all20.append([join_alldata20/2758643,(join_tns_data20)/(351)])
        join_all0.append([join_alldata0/2758643,(join_tns_data0)/(351)])





        join_pns50 = len(np.where(join_pns_data >= (5000))[0])
        join_pns50_ac = join_pns50 / 526

        join_pns_aver =(join_pns_right+join_pns_left)/2
        join_tns_aver=(join_tns_right+join_tns_left)/2
        join_pns_f1 = f1(join_pns50_ac,join_pns_aver)
        join_tns_f1= f1(join_test50_ac,join_tns_aver)

        join_pns_r.append(join_pns_right)
        join_pns_l.append(join_pns_left)
        join_tns_r.append(join_tns_right)
        join_tns_l.append(join_tns_left)
        join_test_ac.append(join_test50_ac)
        join_pns_av.append(join_pns_aver)
        join_tns_av.append(join_tns_aver)
        join_pns_f.append(join_pns_f1)
        join_tns_f.append(join_tns_f1)

#x2data
        X2_alldata = X2[allnum].reshape(1, -1)[0]
        X2_pns_data = X2[pns].reshape(1, -1)[0]
        X2_tns_data = X2[tns].reshape(1, -1)[0]

        X2_alldata_sort = np.sort(np.around(X2_alldata / 1, decimals=0), axis=0)
        X2_pns_data_sort = np.sort(np.around(X2_pns_data / 1, decimals=0), axis=0)
        X2_tns_data_sort = np.sort(np.around(X2_tns_data / 1, decimals=0), axis=0)
        X2_pns_right = AUCC(X2_alldata_sort, X2_pns_data_sort, side="right", flag="X2pnsright:")
        X2_pns_left = AUCC(X2_alldata_sort, X2_pns_data_sort, side="left", flag="X2pnsleft:")
        X2_tns_right = AUCC(X2_alldata_sort, X2_tns_data_sort, side="right", flag="X2tnsright:")
        X2_tns_left = AUCC(X2_alldata_sort, X2_tns_data_sort, side="left", flag="X2tnsleft:")
        #测试概率大于0.5的个数
        X2_test50 = len(np.where(X2_tns_data>=(5000))[0])
        X2_test50_ac = X2_test50/351

        X2_alldata50 = len(np.where(X2_alldata>=(5000))[0])
        X2_alldata50_ac = X2_alldata50/2758643*100

        X2_alldata80 = len(np.where( X2_alldata>=(8000))[0])
        X2_alldata60 = len(np.where( X2_alldata >= (6000))[0])- X2_alldata80
        X2_alldata40 = len(np.where( X2_alldata >= (4000))[0])- X2_alldata80- X2_alldata60
        X2_alldata20 = len(np.where( X2_alldata >= (2000))[0])- X2_alldata80- X2_alldata60- X2_alldata40
        X2_alldata0 = len(np.where( X2_alldata>=0)[0])- X2_alldata80- X2_alldata60- X2_alldata40- X2_alldata20

        X2_pns_data80 = len(np.where(X2_pns_data >= (8000))[0])
        X2_pns_data60 = len(np.where(X2_pns_data >= (6000))[0]) - X2_pns_data80
        X2_pns_data40 = len(np.where(X2_pns_data >= (4000))[0]) - X2_pns_data80 - X2_pns_data60
        X2_pns_data20 = len(np.where(X2_pns_data >= (2000))[0]) - X2_pns_data80 - X2_pns_data60 - X2_pns_data40
        X2_pns_data0 = len(np.where(X2_pns_data >= 0)[0]) - X2_pns_data80 - X2_pns_data60 - X2_pns_data40 - X2_pns_data20

        X2_tns_data80 = len(np.where(X2_tns_data >= (8000))[0])
        X2_tns_data60 = len(np.where(X2_tns_data >= (6000))[0]) - X2_tns_data80
        X2_tns_data40 = len(np.where(X2_tns_data >= (4000))[0]) - X2_tns_data80 - X2_tns_data60
        X2_tns_data20 = len(np.where(X2_tns_data >= (2000))[0]) - X2_tns_data80 - X2_tns_data60 - X2_tns_data40
        X2_tns_data0 = len(np.where(X2_tns_data >= 0)[0]) - X2_tns_data80 - X2_tns_data60 - X2_tns_data40 - X2_tns_data20

        X2_all80.append([X2_alldata80 / 2758643, (X2_tns_data80) / 351])#X2_pns_data80 + X2_pns_data60 +X2_pns_data40 + X2_pns_data20 +X2_pns_data0 +
        X2_all60.append([X2_alldata60 / 2758643, (X2_tns_data60) / 351])
        X2_all40.append([X2_alldata40 / 2758643, (X2_tns_data40) / 351])
        X2_all20.append([X2_alldata20 / 2758643, ( X2_tns_data20) / 351])
        X2_all0.append([X2_alldata0 / 2758643, (X2_tns_data0) / 351])





        X2_pns50 = len(np.where(X2_pns_data>=(5000))[0])
        X2_pns50_ac = X2_pns50/526
        X2_pns50n.append(X2_pns50)

        X2_pns_aver =(X2_pns_right+X2_pns_left)/2
        X2_tns_aver=(X2_tns_right+X2_tns_left)/2
        X2_pns_f1 = f1(X2_pns50_ac,X2_pns_aver)
        X2_tns_f1= f1(X2_test50_ac,X2_tns_aver)
        X2_pns_r.append(X2_pns_right)
        X2_pns_l.append(X2_pns_left)
        X2_tns_r.append(X2_tns_right)
        X2_tns_l.append(X2_tns_left)
        X2_test_ac.append(X2_test50_ac)
        X2_pns_av.append(X2_pns_aver)
        X2_tns_av.append(X2_tns_aver)
        X2_pns_f.append(X2_pns_f1)
        X2_tns_f.append(X2_tns_f1)
        X2_alldatatest50_a.append(X2_alldata50_ac)
print("name",name)
print("join_pns_r",join_pns_r)
print("join_pns_l",join_pns_l)
print("join_tns_r",join_tns_r)
print("join_tns_l",join_tns_l)
print("join_test_ac",join_test_ac)
# print("join_pns_av",join_pns_av)
# print("join_tns_av",join_tns_av)
# print("join_pns_f",join_pns_f)
# print("join_tns_f",join_tns_f)
# print("join_alldata50_a",join_alldata50_a)

print("join_all80",join_all80)
print("join_all60",join_all60)
print("join_all40",join_all40)
print("join_all20",join_all20)
print("join_all0",join_all0)



print("X2_pns_r",X2_pns_r)
print("X2_pns_l",X2_pns_l)
print("X2_tns_r",X2_tns_r)
print("X2_tns_l",X2_tns_l)
print("X2_test_ac",X2_test_ac)
print("X2_pns_av",X2_pns_av)
print("X2_tns_av",X2_tns_av)
print("X2_pns_f",X2_pns_f)
print("X2_tns_f",X2_tns_f)
print("X2_pns50n_ac",X2_pns50n)
print("X2_alldata50_zhanbi",X2_alldatatest50_a)

print("X2_all80",X2_all80)
print("X2_all60",X2_all60)
print("X2_all40",X2_all40)
print("X2_all20",X2_all20)
print("X2_all0",X2_all0)
