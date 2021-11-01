import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from napari.types import ImageData, LayerDataTuple
from napari_plugin_engine import napari_hook_implementation
from osgeo import gdal, gdal_array
from torch.utils.data import random_split

from g_dataset import BasicDataset, imageToDataset

# Hyper Parameters
EPOCH = 2
BATCH_SIZE = 50
LR = 0.01
torch.manual_seed(1)    # reproducible

pre_path = 'E:/BaiduYunDownload/'
# 网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入形式为 batch,features（即通道）,length（文本长度,这里是特征数量）
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=(3,), stride=(1,),padding=1)
        self.pool = nn.MaxPool1d(3)
        self.liner1 = nn.Linear(10 * 2, 2)
    def forward(self, x):
        x = torch.unsqueeze(x,1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 10 * 2)
        x = self.liner1(x)
        return x
# 训练网络
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Mnist
    train_data = BasicDataset('global-sample.csv')
    val_percent = 0.1
    test_percent = 0.2
    n_val = int(len(train_data) * val_percent)
    n_test = int(len(train_data) * test_percent)
    n_train = len(train_data) - n_val - n_test
    train_data, test_data, val_data = random_split(train_data, [n_train, n_test, n_val])
    # 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    cnn = Net()
    print(cnn)  # net architecture

    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    # training and testing
    val_ls = []
    train_ls = []
    test_ls = []

    for epoch in range(EPOCH):
        count = 0
        sum_loss = 0
        for step, da in enumerate(train_loader):  # 分配 batch data, normalize x when iterate train_loader
            b_x = da['image']
            b_y = da['label']
            output = cnn(b_x)  # cnn output
            # print(output)
            # print(b_y)
            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            sum_loss += loss.item()
            count += 1

        # 验证阶段
        val_score = net_test(cnn,val_loader)
        test_score = net_test(cnn,test_loader)
        train_score = net_test(cnn, train_loader)
        val_ls.append(val_score)
        test_ls.append(test_score)
        train_ls.append(train_score)
        print('==================================')
        print('验证集准确率', val_score)
        print('测试集准确率', test_score)
        print('训练集准确率', train_score)
        print('训练误差', sum_loss / count)

    print(val_ls)
    print(test_ls)
    print(train_ls)

    return cnn,test_loader
# 测试网络
def net_test(cnn,test_loader):
    total = 0
    right = 0
    for data in test_loader:
        cnn.eval()
        test_output = cnn(data['image'])
        pred = torch.max(test_output, 1)[1].data.numpy().squeeze()
        for i, label in enumerate(data['label']):
            if label == pred[i]:
                right += 1
            total += 1
    # print(right / total)
    return right / total

import os
from os.path import splitext
from os import listdir
import numpy as np

# 读取数据集
class Landsat8Reader(object):
    def __init__(self,file_path):
        self.base_path = file_path
        self.bands = 7
        self.band_file_name = []

    def read(self):
        tif_files = [os.path.join(self.base_path,f) for f in listdir(self.base_path) if splitext(f)[1]=='.tif']
        tif_file = tif_files[0][:-5]
        for band in range(self.bands):
            band_name = os.path.join(self.base_path,tif_file) + str(band + 1) + ".tif"
            self.band_file_name.append(band_name)
        ds = gdal.Open(self.band_file_name[0])
        image_dt = ds.GetRasterBand(1).DataType
        image = np.zeros((ds.RasterYSize, ds.RasterXSize, self.bands),
                         dtype= \
                             gdal_array.GDALTypeCodeToNumericTypeCode(image_dt))
        for band in range(self.bands):
            ds = gdal.Open(self.band_file_name[band])
            band_image = ds.GetRasterBand(1)
            image[:, :, band] = band_image.ReadAsArray()

        return image

# 计算熵
def generater_entr(cnn,test_loader):
    entr = 0
    for data in test_loader:
        cnn.eval()
        test_output = cnn(data['image'])
        test_output = torch.softmax(test_output,1)
        test_output = test_output[:,0]
        test_output = -test_output*torch.log(test_output)-(1-test_output)*torch.log(1-test_output)
        entr += torch.sum(test_output)
    return entr

# 筛选数据
def one_select(file_name):
    # L开头的数据是land数据
    files = os.listdir(file_name)
    files = [i for i in files if i[0]=='L']

    cnn = Net()
    cnn.load_state_dict(torch.load('./trained_model/net1.pkl'))

    index = 0
    metrc = 0
    for i,file in enumerate(files):
        dataset = imageToDataset(pre_path+file)
        dataset_loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
        entr = generater_entr(cnn,dataset_loader)
        print(i,entr)
        if entr > metrc:
            index = i
            metrc = entr

    selected = files[index]
    return selected

# 主函数
def trainSelect()->LayerDataTuple:
   cnn, test_data = train()
   torch.save(cnn.state_dict(), './trained_model/net1.pkl')
   net_test(cnn, test_data)
   print('训练完毕，开始读取图像')
   pred = one_select(pre_path)
   reader = Landsat8Reader(os.path.join(pre_path,pred))
   image = reader.read()
   index = np.array([4, 3, 2])
   print('读取完毕')
   return ImageData(image[:,:,index]), {'colormap':'turbo'}


# this line is explained below in "Decorating your function..."
@napari_hook_implementation
def napari_experimental_provide_function():
   return trainSelect



