from os.path import splitext
from os import listdir
import os
from osgeo import gdal,gdal_array
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

# if running on a server
OnSercer = False


# 生成训练数据集
class BasicDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        self.labels = data['CLASS'].values
        data = data[['BLUE','GREEN','RED','NIR','SWIR1','SWIR2']]
        m,n = data.shape
        self.lines = []
        for i in range(m):
            self.lines.append(data.loc[i].values)


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        # 获取文件名
        train_data = self.lines[i]

        return {'image':torch.from_numpy(train_data).type(torch.FloatTensor),'label':torch.tensor(self.labels[i]).type(torch.LongTensor)}


# 读取特征
features_index = [2,3,4,5,6,7]
class Landsat8Reader(object):
    def __init__(self,file_path):
        self.base_path = file_path
        self.bands = 6
        self.band_file_name = []

    def read(self):
        tif_files = [os.path.join(self.base_path,f) for f in listdir(self.base_path) if splitext(f)[1]=='.tif']
        tif_file = tif_files[0][:-5]
        for i in features_index:
            self.band_file_name.append(os.path.join(self.base_path,tif_file) + str(i) + ".tif")
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

# 归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return 0.2*(data - np.min(data)) / _range


# 将图片转化为数据集
class imageToDataset(Dataset):
    def __init__(self, file_path):
        reader = Landsat8Reader(file_path)
        image = reader.read()

        # 只取少量像素
        image = normalization(image[:,3500:3502,:])
        m,n,l = image.shape

        self.lines = []
        for i in range(m):
            for j in range(n):
                self.lines.append(image[i,j,:])

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):
        # 获取文件名
        train_data = self.lines[i]
        return {'image':torch.from_numpy(train_data).type(torch.FloatTensor)}

