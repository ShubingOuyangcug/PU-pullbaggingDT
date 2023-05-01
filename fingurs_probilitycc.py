
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time

from networks25 import EmbeddingNet
from sklearn import preprocessing
import gdal
import numpy as np
import os
os.environ['PROJ_LIB'] = r'/home/cv/anaconda3/envs/test/share/proj'
device = torch.device('cuda:0')
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
    x1 = x[s2<3]
    # X_mean = x1.mean(axis=0)
    # X_std = x1.std(axis=0)
    # X22 = (x - X_mean) / X_std
    scaler = preprocessing.StandardScaler().fit(x1)
    x = scaler.transform(x)

    # max_index = np.unravel_index(np.argmax(x, axis=None), x.shape)#获取数据最大值及索引位置
    # max_value = x[max_index]
    # print(max_index, max_value)
    del in_ds
    return x

in_ds = gdal.Open(r'./newlabel2/label54.tif')  # 读取要切的原图
# print("open tif file succeed")
width = in_ds.RasterXSize  # 获取数据宽度
print(width)
height = in_ds.RasterYSize
s = in_ds.ReadAsArray(0, 0, width, height)  # 获取数据.astype(np.float)
# get the tif parameters
im_geotrans = in_ds.GetGeoTransform()  # 获取仿射矩阵信息
im_proj = in_ds.GetProjection()
s2 = s.flatten()
del in_ds
foldpath = r'./newlast6'
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

def setDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    :param filepath:需要创建的文件夹路径
    :return:
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    # else:
    #     shutil.rmtree(filepath,ignore_errors=True)





def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 25*2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:

            #     images = torch.tensor(np.array(images))
            images = images.to(device)
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            # embeddings[k:k + len(images)] = model.get_embedding(images).data.cpu()
            # labels[k:k + len(images)] = target
            k += len(images)
    return embeddings, labels

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.train_data = torch.tensor(data, dtype=torch.float32)
        self.train_labels = torch.tensor(label, dtype=torch.long)
        self.test_data = torch.tensor(data, dtype=torch.float32)
        self.test_labels = torch.tensor(label, dtype=torch.long)



    def __len__(self):
        return len(self.train_data )

    def __getitem__(self, index):
        dataall = self.train_data[index]
        target  = self.train_labels[index]
        return dataall, target

model = EmbeddingNet()
path_name2 = "modelsave"
path1 =r'./result_picture/'+path_name2
path =  r'./'+path_name2+'/'  #./H73/0.9_0_1_46.h5./H96/0.4_4_4_122.h5 ./H91/0_0_1_832.h5./H115/0_0_8_2362.h5/H116/0_0_4_1262.h5
setDir(path1)
epoch = 152 #937/H54/0.4_1_3_499.h5./H73/0_0_2_43.h5./H79/0_0_1_114.h5./H109/0_4_2_2032.h5H111/0_0_3_4992.h50_0_5_1862.h5,"0_2_7_92.h5","0_2_8_1902.h5","0_2_9_1262.h5"
for ipath_name in ['1_4_10_2992.h5']:
    path_name = ipath_name#s255'0_5_7_12.h5','0_5_6_12.h5','0_5_5_12.h5','0_5_4_12.h5','0_5_3_12.h5','0_5_2_12.h5','0_5_1_32.h5','0_5_10_12.h5','0_4_9_12.h5','0_4_8_52.h5','0_4_7_12.h5','0_4_6_12.h5','0_4_5_12.h5','0_4_4_52.h5','0_4_3_12.h5','0_4_2_12.h5','0_4_1_42.h5','0_4_10_12.h5','0_2_9_72.h5','0_2_8_12.h5','0_2_7_12.h5','0_2_6_12.h5','0_2_5_12.h5','0_2_4_22.h5','0_2_3_82.h5','0_2_1_32.h5','0_2_10_12.h5']:

    model_path = os.path.join(path, path_name)  # "method_dataset_name_" + str(i) + ".h5"./H59/0_0_6_54.h5./H103/0_0_3_2332.h5H104/0_4_2_2032.h5

    trained_model = os.path.join(model_path)
    load_net(trained_model, model)

    if torch.cuda.is_available():
        net = model.to(device)

    batchsize = 5000

    train_dataset = Dataset (X, s2)

    print(len(train_dataset))

    train_loader =torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batchsize,pin_memory=True, num_workers=8)


    train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
    probility = []
    iter_number = 0


    label_embeddings = torch.tensor(train_embeddings_cl[s2 == 2])##5
    label_embeddings =label_embeddings.type(torch.float32)
    print(label_embeddings.type())
    all_result = torch.tensor(train_embeddings_cl[s2 <6])

    llens = len(label_embeddings)




    train_dataset2 = Dataset (train_embeddings_cl, train_labels_cl)

    allsum=0

    print("start")

    for result, labels in train_dataset2:
        if labels<3:
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            result = result.reshape(1,25*2)

            output = cos(result, label_embeddings)
            positive_dis = (output.mean() + 1)/2*100000

            positive_dis = positive_dis.type(torch.int32)
        else:
            positive_dis=torch.tensor(1,dtype=torch.int32)


        probility.append(positive_dis)
    print("startsave")
    probility = np.around(np.array(probility)/10, decimals=0)

    print(type(probility))

    band = 1
    probility = probility.reshape(height, width)
    print(probility.shape)
    newpath= r'./result_picture/'+path_name2+'/' +path_name2+path_name+'.tif'
    print(newpath)
    write_tif(newpath, probility, im_geotrans, im_proj, width, height, gdal.GDT_Int16)

