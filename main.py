import torch
import h5py
import torch.nn.functional as F
import numpy as np
import cv2
import nibabel as nib
import dataLoader
import cupyx.scipy.ndimage as filter
import cupy as cp
import os
import sys

from tqdm import tqdm
dirs = cp.loadtxt("./45.txt")[1:,:-1].astype(np.float32)
def split(data, pad, block,roi = None):
    pad = pad
    block = block
    shape = data.shape
    if roi is None:
        roi = [0,shape[2],0,shape[0],0,shape[1]]
    flag = [[i for i in range(roi[j*2], roi[j*2+1], block[j])] for j in range(3)]
    for i in range(len(flag)):
        flag[i].append(roi[i*2+1])
    pad_index = []
    effective_index = []
    ODF_Index = []
    for i in range(len(flag[0]) -1):
        for j in range(len(flag[1]) -1):
            for k in range(len(flag[2])-1 ):
                xl = max(flag[0][i] - pad, 0);xr = min(flag[0][i + 1] + pad, roi[1]);yl = max(flag[1][j] - pad, 0);yr = min(flag[1][j + 1] + pad, roi[3]);zl = max(flag[2][k] - pad, 0);zr = min(flag[2][k + 1] + pad, roi[5])
                ODF_Index.append(
                    [[flag[0][i], flag[0][i + 1]], [flag[1][j], flag[1][j + 1]], [flag[2][k], flag[2][k + 1]]])
                pad_index.append([[xl, xr], [yl, yr], [zl, zr]])
                effective_index.append([[flag[0][i] - xl, flag[0][i + 1] - xl],
                                       [flag[1][j] - yl, flag[1][j + 1] - yl],
                                       [flag[2][k] - zl, flag[2][k + 1] - zl]])
    return pad_index, effective_index, ODF_Index

def grad(temp):
    Vx = filter.gaussian_filter(temp, sigma * 1, order=[1, 0, 0], mode='nearest', truncate=4)[::ratio, ::ratio,::ratio]
    Vy = filter.gaussian_filter(temp, sigma * 1, order=[0, 1, 0], mode='nearest', truncate=4)[::ratio, ::ratio,::ratio]
    Vz = filter.gaussian_filter(temp, sigma * 1, order=[0, 0, 1], mode='nearest', truncate=4)[::ratio, ::ratio,::ratio]
    return Vx, Vy, Vz
def psf(gradient, index):
    gradient_norm = np.linalg.norm(gradient, axis=-1)[...,None]
    gradient = np.nan_to_num(gradient / gradient_norm)
    o = gradient @ dirs.T
    return o ** index * gradient_norm

sigma = 1
rho = 20
ratio = 2

def main():
    address = sys.argv[1]
    h5 = h5py.File(         address,        'r')
    img = h5['DataSet']['ResolutionLevel 2']['TimePoint 0']['Channel 0']['Data']
    ims_roi = [0,img.shape[0],
               0,img.shape[1],
               0,img.shape[2]]
    nii_roi = [img.shape[2]-ims_roi[5], img.shape[2]-ims_roi[4],
               img.shape[0]-ims_roi[1] , img.shape[0]-ims_roi[0],
               img.shape[1]-ims_roi[3], img.shape[1]-ims_roi[2]]
    ODF_Data = np.zeros([(nii_roi[1]-nii_roi[0]) // rho,
                      (nii_roi[3]-nii_roi[2]) // rho,
                      (nii_roi[5]-nii_roi[4]) // rho, 46], dtype=np.float32)
    padded_index, effective_index, ODF_Index= split(img, 1 * rho, [10 * rho, 10 *  rho, 10 * rho],nii_roi)
    dataSet = dataLoader.CustomIMSDataset(padded_index, effective_index, ODF_Index,address)
    dataloader = dataLoader.DataLoader(dataSet,batch_size=1,num_workers=10,pin_memory=True)
    k = torch.tensor(cv2.getGaussianKernel(rho//ratio + 1, rho//ratio*10, cv2.CV_32F)[:, 0]).cuda(0)
    i = 0
    for batch in tqdm(dataloader, colour = 'GREEN'):
        roi = cp.asarray(batch[0][0])
        flag = cp.asarray(batch[1][0])[::ratio, ::ratio,::ratio]

        Vx, Vy, Vz = grad(roi)
        Vx[flag]=0; Vy[flag]=0; Vz[flag]=0
        r = cp.stack(( Vx, Vy, Vz), axis=-1)
        x = psf(r,100)
        x = torch.as_tensor(x[None, ...].transpose([4, 0, 1, 2, 3])).cuda()
        x = F.conv3d(x, k[None, None, None, None, :], padding=(0, 0, k.shape[0] // 2), stride=[1, 1, rho // ratio])
        x = F.conv3d(x, k[None, None, None, :, None], padding=(0, k.shape[0] // 2, 0), stride=[1, rho // ratio, 1])
        x = F.conv3d(x, k[None, None, :, None, None], padding=(k.shape[0] // 2, 0, 0), stride=[rho // ratio, 1, 1])
        x = cp.array(x[..., None].transpose(0, -1)[0][0])
        cp.nan_to_num(x, 0)
        x = cp.concatenate([cp.max(x[:], axis=-1)[..., None] * 2, x], axis=-1)
        x[...,0] = cp.where(x[...,0]>0.2,1,0)
        x=x.get()
        odf = ODF_Index[i]
        eff = effective_index[i]
        ODF_Data[ (odf[0][0]-nii_roi[0]) // rho:(odf[0][1]-nii_roi[0]) // rho,(odf[1][0]-nii_roi[2]) // rho:(odf[1][1]-nii_roi[2]) // rho,(odf[2][0]-nii_roi[4]) // rho:(odf[2][1]-nii_roi[4]) // rho] = x[eff[0][0] // rho:  eff[0][1] // rho,  eff[1][0] // rho: eff[1][1] // rho,  eff[2][0] // rho: eff[2][1] // rho]
        i += 1

    out = makeNiiForMRView(rho, ODF_Data)
    dwi = f'./DWI_sigma_{sigma}_rho_{rho}.nii'
    res = f'./response1.nii'
    fod = f'./FOD_sigma_{sigma}_rho_{rho}.nii'
    dir_file = f'./45.txt'
    nib.save(out, dwi)

    # os.system(f'dwi2response tax {dwi}  {res}  -grad {dir_file}  -force')
    os.system(f'mrfilter  {dwi} smooth -stdev 0.06,0.06,0.06 {dwi}   -force')
    os.system(f'dwi2fod msmt_csd {dwi} {res} {fod} -grad {dir_file} -lmax 8 -shells 1000 -force')
    os.system(f'mrview  {dwi} -imagevisible false -odf.load_sh {fod}')

def makeNiiForMRView(fod_reslotion, out):
    out = nib.Nifti1Image(out, np.eye(4) * fod_reslotion*4/1000)
    out.affine[0, -1] = fod_reslotion / 2
    out.affine[1, -1] = fod_reslotion / 2
    out.affine[2, -1] = fod_reslotion / 2
    return out

if __name__ == '__main__':
    main()
