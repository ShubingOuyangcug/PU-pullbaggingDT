from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn import datasets
from sklearn import preprocessing
import time
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import gdal
import os

os.environ['PROJ_LIB'] = r'/home/cv/anaconda3/envs/test/share/proj'


ospath = r'./result_picture/modelsave'#
name = ospath+"guiyi"



in_ds = gdal.Open(r'./newlabel2/label54.tif')  # 读取要切的原图
# print("open tif file succeed")
width = in_ds.RasterXSize  # 获取数据宽度
height = in_ds.RasterYSize
im_geotrans = in_ds.GetGeoTransform()  # 获取仿射矩阵信息
im_proj = in_ds.GetProjection()
s = in_ds.ReadAsArray(0, 0, width, height)  # 获取数据.astype(np.float)
s2 = s.flatten()
del in_ds

def setDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    :param filepath:需要创建的文件夹路径
    :return:
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)

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

def read(path,s2):
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
    x[s2 == 3] = x.max()
    print(x.max())
    x1 = x[s2<3]


    # X_mean = x1.mean(axis=0)
    # X_std = x1.std(axis=0)
    # X22 = (x - X_mean) / X_std
    min_max_scaler = preprocessing.MinMaxScaler()
    x2 = min_max_scaler.fit_transform(x)
    print(x2)
    return np.around((x2*10000),decimals=0)


for i in os.listdir(ospath):
    if "tif" in i:
        path2 = ospath +'/' + i
        X2 = read(path2,s2)
        print("startsave")
        probility = np.around(np.array(X2) , decimals=0)

        print(type(probility))

        probility = probility.reshape(height, width)
        pathpic = name
        setDir(pathpic)
        print(probility.shape)
        newpath = name + '/' + i
        print(newpath)
        write_tif(newpath, probility, im_geotrans, im_proj, width, height, gdal.GDT_Int16)