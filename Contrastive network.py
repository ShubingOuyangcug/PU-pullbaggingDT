
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
from evaluate_model import evaluate_model
from sklearn.model_selection import train_test_split
os.environ['PROJ_LIB'] = r'/home/cv/anaconda3/envs/test/share/proj'
from sklearn import preprocessing
import gdal
import numpy as np
import os

# reading tif images
def read(path,s2):
    in_ds = gdal.Open(path)
    print("open tif file succeed")
    width = in_ds.RasterXSize
    height = in_ds.RasterYSize
    outbandsize = in_ds.RasterCount
    im_geotrans = in_ds.GetGeoTransform()
    im_proj = in_ds.GetProjection()
    datatype = in_ds.GetRasterBand(1).DataType
    x = in_ds.ReadAsArray(0,0,width,height)
    x = x.reshape(-1, 1)
    x1 = x[s2<3]
    scaler = preprocessing.StandardScaler().fit(x1)
    x = scaler.transform(x)
    del in_ds
    return x,im_geotrans,im_proj


# save the model
def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


# the network of Discriminator is the Deep Learning Architecture
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # the channel is the number of input channels
        channel = 25
        self.fc = nn.Sequential(
            nn.Linear(channel, channel*2),
            nn.ReLU(inplace=True),
            nn.Linear(channel*2,channel*4),
            nn.ReLU(inplace=True),
            nn.Linear(channel*4, channel*2),
            nn.ReLU(inplace=True),
            nn.Linear(channel*2, channel*2),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x = self.fc(x)
        return F.normalize(x, dim=-1)

    def get_embedding(self, x):
        return self.forward(x)

def normalize(x, axis=1):
    x = 1.0 * x / (torch.norm(x, 2, axis, keepdim=True) + 1e-12)
    return x


def infiniteloop(dataloader):
    while True:
        for x in iter(dataloader):
            yield x


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def setDir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)

def main(tau_plusx, epochnum, re,randonKF):
    tau_plus = tau_plusx
    # label54.tif is the label
    in_ds = gdal.Open(r'./newlabel2/label54.tif')
    width = in_ds.RasterXSize
    height = in_ds.RasterYSize
    s = in_ds.ReadAsArray(0, 0, width, height)
    s2 = s.flatten()
    del in_ds
    # foldpath is fold of input factors
    foldpath = r'./newlast6'
    n=0
    for name in os.listdir(foldpath):
        if name[-3:]=="tif":
            path = foldpath + "/" +name
            print(path)
            if n==0:
                X,im_geotrans,im_proj = read(path,s2)
                n = n+1
            else:
                out,im_geotrans,im_proj = read(path,s2)
                X = np.concatenate((out, X), axis=1)

    # train_dataset is testing set and unknown landslides cells
    train_dataset = Dataset(X[s2 < 2])

    # twotest_dataset is the training set and validation set
    twotest_dataset = Dataset (X[s2 == 2])
    print(len(twotest_dataset))

    # testdata_loader is the testing set
    testdata_loader = Dataset (X[s2 == 1])

    num = len(train_dataset)
    print(num)

    print("test:",len(testdata_loader))
    kf = KFold(n_splits=10, shuffle=True, random_state=randonKF)

    splitci =0

    for train_index, test_index in kf.split(twotest_dataset):
        jufu52, jufu50, jufu47 = 10000000000, 10000000000, 10000000000
        splitci+=1
        if len(train_index)%2 ==0:
            train_all_index = train_index
            test_all_index = test_index
            two_train = twotest_dataset[train_index]
            two_valid = twotest_dataset[test_index]
        else:
            train_all_index = np.append(train_index,test_index[0])
            test_all_index =test_index[1:]
            two_train = twotest_dataset[train_all_index]
            two_valid = twotest_dataset[test_all_index]
        print("TRAIN:", train_all_index, "TEST:", test_all_index)
        print("TRAIN:", len(train_all_index), "TEST:", len(test_all_index))



        test_loader =torch.utils.data.DataLoader(dataset=testdata_loader,batch_size=len(testdata_loader),pin_memory=True, num_workers=8)
        unlabeled_loader =torch.utils.data.DataLoader(dataset=train_dataset,batch_size=350000,pin_memory=True,shuffle=True, drop_last=True, num_workers=8)
        labeled_loader =torch.utils.data.DataLoader(dataset=two_train,batch_size=int(len(two_train)),pin_memory=True,shuffle=True, drop_last=True, num_workers=8)
        valid_labeled_loader =torch.utils.data.DataLoader(dataset=two_valid,batch_size=len(two_valid),pin_memory=True,num_workers=8)
        all_label =torch.utils.data.DataLoader(dataset=two_train,batch_size=len(two_train),pin_memory=True, num_workers=8)



        device = torch.device('cuda:0')
        encoding = Discriminator().to(device)
        lr = 1e-4
        optimizer = torch.optim.Adam(encoding.parameters(), lr=lr)


        labeled_looper = infiniteloop(labeled_loader)
        num_epoch = epochnum

        rand_seed = 64678
        if rand_seed is not None:
            np.random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed(rand_seed)

        record=[]
        for epoch in range(num_epoch):
            loss = 0
            encoding.zero_grad()

            for unlabeled_x in iter(unlabeled_loader):
                xp = next(labeled_looper).to(device)
                xu = unlabeled_x.to(device)
                xall = torch.cat([xp,xu],dim=0)
                outall = encoding(xall)
                batchp = int(len(two_train))
                outxp = outall[:batchp]
                outxanchor = outxp
                outxu = outall[batchp:]

                # LOSS: sum(zi*zp)
                posin = torch.mm(outxp, outxanchor.t().contiguous())
                negc =torch.mm(outxp, outxu.t().contiguous())
                posindiag = torch.diag(posin)
                a_diag = torch.diag_embed(posindiag)
                posin = posin - a_diag

                # LOSS: exp(sum(zi*zp))
                pos = torch.exp(torch.sum(posin, dim=-1) / ((len(train_all_index) - 1) * 1))

                # LOSS: exp(sum(exp(zi*zu)))
                neg = torch.sum(torch.exp(negc/1), dim=-1)

                output1 = (- torch.log(pos / (pos + neg))).mean()
                output = output1
                lossall = output
                loss += lossall
                encoding.zero_grad()
                lossall.backward()
                optimizer.step()

            if epoch % 1 == 0:
               record1 = (f'[{epoch}/{num_epoch}]: Loss={loss}')
               print(record1)
            # save net
            if epoch % 1 == 0:
                encoding.eval()
                flag = 0
                flag0 = 0
                flag2 = 0
                flag3 = []
                flag5 = []
                path_name2 = "modelsave"
                output_dir = r'./' + path_name2 + '/'
                setDir(path_name2)

                re =randonKF
                save_name = os.path.join(output_dir, '{}_{}_{}_{}2.h5'.format(str(tau_plus), str(re), str(splitci) ,str(epoch))) # "method_dataset_name_" + str(i) + ".h5"
                save_net(save_name, encoding)


                for all_label_x in iter(all_label):
                    for valid_label_x in iter(valid_labeled_loader):
                        for test_loader_x in iter(test_loader):

                            all_label_x = all_label_x.to(device)
                            valid_label_x = valid_label_x.to(device)
                            test_loader_x = test_loader_x.to(device)
                            xall22 = torch.cat([all_label_x, valid_label_x, test_loader_x], dim=0)
                            xallout2 = encoding(xall22)
                            outtrain = xallout2[:len(all_label_x)]
                            outvalid = xallout2[len(all_label_x):(len(valid_label_x)+len(all_label_x))]
                            train_all = xallout2[:(len(valid_label_x)+len(all_label_x))]
                            outtest = xallout2[(len(valid_label_x)+len(all_label_x)):]

                            #Distance
                            for ouut in iter(unlabeled_loader):
                                ouut= ouut.to(device)
                                ouut = encoding(ouut)
                                for iii in range(outvalid.shape[0]):
                                    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                                    result = outvalid[iii].reshape(1, channel1 * 2)
                                    distance0 = cos(result, ouut).mean()
                                    flag = flag + distance0

                            # similarA
                            for iii in range(outvalid.shape[0]):
                                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                                result = outvalid[iii].reshape(1,channel1*2)
                                output = cos(result, train_all)
                                distance = output.mean()
                                flag0 = flag0 + distance
                                flag5.append(distance.item())

                            for iiii in range(outtest.shape[0]):
                                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                                result2 = outtest[iiii].reshape(1, channel1*2)
                                output2 = cos(result2, train_all)
                                distance2 = output2.mean()
                                flag3.append(distance2.item())
                                flag2 = flag2 + distance2
                flag0 = flag0/(len(test_all_index))
                flag2 = flag2 / (outtest.shape[0])
                flag = flag             #
                jfu = flag.item()

                L=0
                L2 = 0
                for numm in flag3:
                    if numm > 0:
                        L = L + 1

                # N similarA>0
                for numm2 in flag5:
                    if numm2 > 0:
                        L2 = L2 + 1

                # set threshold
                if L2 == 52:
                    if jfu < jufu52:
                        jufu52 = jfu
                        min152 = flag0
                        epoch52 = epoch
                        L52=L
                        save_name52 = save_name

                if L2 == 50:
                    if jfu < jufu50:
                        jufu50 = jfu
                        min150 = flag0
                        epoch50 = epoch
                        L50 = L
                        save_name50 = save_name

                if L2 == 47:
                    if jfu < jufu47:
                        jufu47 = jfu
                        min147 = flag0
                        epoch47 = epoch
                        L47 = L
                        save_name47 = save_name

                record2 = f'[{epoch}/{num_epoch}]: -------------------------validLoss={flag0.item(), flag.item()},l={L},jufu={jfu}'
                print(record2)

                record3 = (
                    f'[{epoch}/{num_epoch}]: --------#{save_name}#--------------------------------testLoss={flag2}-L={L}-{flag3}')
                print(record3)  # -{flag3}
                record5 = (
                    f'[{epoch}/{num_epoch}]: --------#{save_name}#---------------------------------------------L2={L2}---{flag5}')
                print(record5)
                record.append(record1)
                record.append(record2)
                record.append(record3)
                record.append(record5)

        record52 =(
                    f'jufu52#{jufu52}#{L52}#{min152}#{epoch52}#{save_name52.split("/")[1]}#"{save_name52.split("/")[2]}",')
        print(record52)
        try:
            record50 =(
                    f'jufu50#{jufu50}#{L50}#{min150}#{epoch50}#{save_name50.split("/")[1]}#"{save_name50.split("/")[2]}",')
            print(record50)
        except:
            pass
        try:
            record47 =(
                    f'jufu47#{jufu47}#{L47}#{min147}#{epoch47}#{save_name47.split("/")[1]}#"{save_name47.split("/")[2]}",')
            print(record47)
        except:
            pass


        txtname = "AGlid20116" + str(re)+str(epoch) + "newu19" + output_dir.replace("/", "") + "T" + str(
            tau_plus) + "ci"+str(splitci)+"_xun2.txt"
        try:
            with open(txtname, "w") as f:
                f.write(str(time.localtime()) + "\n")
                f.write("T:" + str(str(tau_plus)) + "\n")
                f.write(output_dir + str(epoch) + "\n")
                for numb in record:
                    f.write(str(numb) + "\n")
        except:
            pass
        txtname2 = "Duandian" + str(re)+str(epoch) + "newu19" + output_dir.replace("/", "") + "T" + str(
            tau_plus) + "ci"+str(splitci)+"_xun2.txt"

        with open(txtname2, "w") as f:
            f.write(str(time.localtime()) + "\n")
            f.write("T:" + str(str(tau_plus)) + "\n")
            f.write(output_dir + str(epoch) + "\n")
            f.write(str(record52) + "\n")
            try:
                f.write(str(record50) + "\n")
            except:
                pass
            try:
                f.write(str(record47) + "\n")
            except:
                pass

# the channel1 is the number of input channels
channel1 =25
# re parameter has no meaning
re = 1
epochnum = 60
# tau_plusx parameter has no meaning
for tau_plusx in [1]:
    # randonKF is the random number that is set
    for randonKF in [5, 4, 2, 8, 9]:
        main(tau_plusx, epochnum, re, randonKF)


