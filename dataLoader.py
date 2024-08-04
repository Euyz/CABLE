import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import h5py

class CustomIMSDataset(Dataset):
    def __init__(self, ODF_partition, data, z0=0):
        self.ODF_partition=ODF_partition
        self.data=data
        self.open_hdf5()
        self.z0=z0
        # self.flag=np.zeros(self.data.shape,dtype=bool)
        # self.flag[0::75]=True
        # self.flag[1::75] = True
        # self.flag[2::75] = True
        # self.flag[74::75] = True
        # self.flag[73::75] = True

    def getDataByNiiIndexFromTif(self, tif, nii_roi):
        temp=self.axisTransform(tif,nii_roi)

        return torch.tensor(temp.astype(np.float32))

    def axisTransform(self,tif,nii_roi):
        shape = tif.shape
        temp = tif[shape[0] - nii_roi[1][1]: shape[0] - nii_roi[1][0],
               shape[1] - nii_roi[2][1]:shape[1] - nii_roi[2][0], shape[2] - nii_roi[0][1]:shape[2] - nii_roi[0][0]]
        temp = np.flip(temp, axis=(0, 1, 2))
        temp = np.swapaxes(temp, 0, 1)
        temp = np.swapaxes(temp, 0, 2)
        return temp

    def __len__(self):
        return len(self.ODF_partition)

    def open_hdf5(self):
        self.img_hdf5 = h5py.File(self.data, 'r')
        self.data = self.img_hdf5['DataSet']['ResolutionLevel 2']['TimePoint 0']['Channel 1']['Data']

    def __getitem__(self, i):
        if not hasattr(self, 'img_hdf5'):
            self.open_hdf5()
        return self.getDataByNiiIndexFromTif(self.data, self.ODF_partition[i]),self.ODF_partition[i]
