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
def write_tif(newpath,im_data,im_Geotrans,im_proj, width, height, datatype):
    if len(im_data.shape)==3:
        im_bands, im_height, im_width = im_data.shape
        print(im_data.shape)

    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    im_width = width
    im_height = height
    im_bands = 1
    diver = gdal.GetDriverByName('GTiff')
    new_dataset = diver.Create(newpath, im_width, im_height, im_bands, datatype)
    new_dataset.SetGeoTransform(im_Geotrans)
    new_dataset.SetProjection(im_proj)

    if im_bands == 1:
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        new_dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            new_dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del new_dataset


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
    print(len(re2))
    for iii in range(1, 101):
        fflag = 100 - iii
        re3 = np.where(np.logical_and(re2 <= (100 / 100), re2 >= (fflag / 100)))
        nnum = nnum + len(re3[0])
        jilu = str(fflag + 1) + ":" + str(nnum)
        numhe.append(jilu)
    print(nnum)
    print(flag,nnum / (len(re2) * 100))
    print(numhe)
outputname = "jufu12_52_5"
path = r'/usr/ouyangyx/biye12/result_picture/modelsaveguiyi/modelsave1_5_10_532.h5.tif' #1_5_10_112.h5,1_5_6_82.h5,e25520Eaver.tifresult691000real150.tifsum251bag253.tifsum15.tif H250/e25n0_2_5_92.h5.tif sumn25.tif aaalllb32 result691000real.tif result69_0006 re2.tif resultadapu(122013,).tifresultadapu(3879487,).tif result691000real17.tifresult691000real150.tif
# print(path)
X1 = read(path)
path2 = r'/usr/ouyangyx/biye12/result_picture/modelsaveguiyi/modelsave1_5_2_372.h5.tif'
X2 = read(path2)
X=(np.around(X1/1, decimals=0)+np.around(X2, decimals=0))/2

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

print(X)
allnum =len(s2)
anll =  np.arange(allnum)
pns = anll[s2 == 2]
tns = anll[s2 == 1]
allnum = anll[s2 < 3]
pns_data = X[pns].reshape(1,-1)[0]
alldata =X[allnum].reshape(1,-1)[0]
alldata50 = len(np.where(alldata>=5000)[0])

probility1 = np.around(X, decimals=0)
probility =probility1.reshape(height, width)
print(probility.shape)
newpath= r'/usr/ouyangyx/biye12/result_picture/he/'+outputname+'.tif'
print(newpath)
write_tif(newpath, probility, im_geotrans, im_proj, width, height, gdal.GDT_Int16)

# print("x2_tns_data_sort")
# x2_tns_data=X2[tns].reshape(1,-1)[0]
# x2_tns_data_sort = np.sort(np.around(x2_tns_data/1, decimals=0),axis = 0)
# print(x2_tns_data_sort )
# x2_alldata_sort = np.sort(np.around(X2[allnum].reshape(1,-1)[0]/1, decimals=0),axis = 0)
# AUCC(x2_tns_data_sort, x2_tns_data, side="right",flag="testright:")
#
# tnsdata=X[tns].reshape(1,-1)[0]
# alldata_sort = np.sort(np.around(alldata/1, decimals=0),axis = 0)
# pns_data_sort = np.sort(np.around(pns_data/1, decimals=0),axis = 0)
# print("pns_data_sort")
# print(pns_data_sort)
# tnsdata_sort = np.sort(np.around(tnsdata/1, decimals=0),axis = 0)
# print("tnsdata_sort")
# mmm = len(np.where(tnsdata<5000)[0])
# print(mmm)
# print(tnsdata_sort)
# print(alldata_sort)
# re = np.searchsorted(alldata_sort,tnsdata_sort, side='right')#rightleft
# print(re)
# re2 = re/2758643
# print(re2)
# print("--------------------------")
#
# ssum=0
# nnum =0
# print(len(re2))
# for iii in range(1,101):
#     fflag = 100-iii
#     re3 = np.where(np.logical_and(re2<=(100/100),re2>=(fflag/100)))
#
#     nnum= nnum + len(re3[0])
#
# alldataX1 =X1[allnum].reshape(1,-1)[0]
# alldataX150 = len(np.where(alldataX1>=5000)[0])
# tnsdataX1=X1[tns].reshape(1,-1)[0]
#
# alldata_sortX1 = np.sort(np.around(alldataX1/1, decimals=0),axis = 0)
#
# tnsdata_sortX1 = np.sort(np.around(tnsdataX1/1, decimals=0),axis = 0)
#
# AUCC(alldata_sortX1, tnsdata_sortX1, side="right",flag="testright:")
# print(nnum)
# print(nnum/(len(re2)*100))
# print(re2.sum())
# print(mmm)
# print((351-mmm)/351*100)
# print("small_0.5zhanbi",(mmm)/878*100)
# print(path,path2)
# print("pubagging_alldata_big50",alldataX150/2758643*100)
# print("jionalldata_big50",alldata50/2758643*100)