
from __future__ import division, print_function
import numpy as np

import matplotlib.pylab as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
import sklearn
from sklearn.metrics import precision_recall_curve
from sklearn import preprocessing
import gdal
import numpy as np
import os
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
from evaluate_model import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
os.environ['PROJ_LIB'] = r'/home/cv/anaconda3/envs/test/share/proj'
cuda = torch.cuda.is_available()

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
    x1 = x[s2<3]
    scaler = preprocessing.StandardScaler().fit(x1)
    x = scaler.transform(x)

    del in_ds
    return x

def write_tif(newpath,im_data,im_Geotrans,im_proj, width, height, datatype):
    if len(im_data.shape)==3:
        im_bands, im_height, im_width = im_data.shape
        print(im_data.shape)
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
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

in_ds = gdal.Open(r'./newlabel2/label54.tif')  # 读取要切的原图
# print("open tif file succeed")
width = in_ds.RasterXSize  # 获取数据宽度
height = in_ds.RasterYSize
# get the tif parameters
im_geotrans = in_ds.GetGeoTransform()  # 获取仿射矩阵信息
im_proj = in_ds.GetProjection()
s = in_ds.ReadAsArray(0, 0, width, height)  # 获取数据.astype(np.float)
s2 = s.flatten()
del in_ds

foldpath = r'./newlast62'
n=0
for name in os.listdir(foldpath):
    if name[-3:]=="tif":
        path = foldpath + "/" +name
        print(path)
        if n==0:
            X = read(path,s2)
            n = n+1
        else:
            out = read(path,s2)
            X = np.concatenate((out, X), axis=1)




allnum =len(X)
pns = np.arange(allnum)[s2 == 2]
uns = np.arange(allnum)[s2==0]#100
tns = np.arange(allnum)[s2 == 1]
allll = np.arange(allnum)[s2 < 3]#100
utns = np.hstack((uns,tns))



data_U = X[utns, :]


TT= [1000]



data_P= X[pns, :]

for T in TT:
    NP = len(data_P)
    K = NP
    train_label = np.zeros(shape=(NP+K,))
    train_label[:NP] = 1
    print(train_label)
    n_oob = np.zeros(shape=(allnum,))
    f_oob = np.zeros(shape=(allnum, 2))
    class_weight = {1: 1, 0: 18}
    for i in range(T):
        # print(i)
        # Bootstrap resample
        bootstrap_sample = np.random.choice(utns, replace=False, size=K)
        # Positive set + bootstrapped unlabeled set

        # print(bootstrap_sample)

        data_bootstrap = np.concatenate((data_P, X[bootstrap_sample, :]), axis=0)
        # Train model
        model = DecisionTreeClassifier(max_depth=None, max_features=None,criterion='gini', class_weight='balanced')
        #model = sklearn.svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,degree=3, gamma='auto', kernel ='rbf',max_iter = -1, probability = True, random_state = None, shrinking = True,tol = 0.001, verbose = False)

        model.fit(data_bootstrap, train_label)
        # Index for the out of the bag (oob) samples
        idx_oob = sorted(set(allll) - set(bootstrap_sample))
        f_oob[idx_oob] += model.predict_proba(X[idx_oob,:])
        n_oob[idx_oob] += 1

        rere = np.where(n_oob == 0)


        predict_proba = (f_oob[:, 1]/n_oob) * 10000

        resulttp = predict_proba[pns]

        where_rererep = np.where(resulttp > 5000)
        txt6 = "positive>50," + str(len(resulttp[where_rererep]))

        resultt = predict_proba[tns]

        where_rerere = np.where(resultt > 5000)

        txt2=str(i)+"epoch,test>50,"+str(len(resultt[where_rerere]))

        resultt3 = predict_proba[predict_proba> 5000]
        txt4= "len>50," + str(len(resultt3))
        txtall = "------------------------------------------------------------------------------------------------" +","+txt2+","+","+txt4+","+txt6
        print(txtall)



        band = 1
        predict_proba = predict_proba.reshape(height, width)
        print(predict_proba.shape)
print(resultt)
newpath= r'./result_picture/'+str(T)+"real1501.tif"
print(newpath)
write_tif(newpath, predict_proba, im_geotrans, im_proj, width, height, gdal.GDT_Int16)