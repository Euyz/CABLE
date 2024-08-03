import os
import sys

import numpy as np
import h5py
import nibabel as nib
import cv2

import torch
import torch.nn.functional as F
import cupyx.scipy.ndimage as filter
import cupy as cp
from tqdm import tqdm

import dataLoader

def split(data, pad, block, roi = None):
    """
    @param
    pad   : padding to block, e.g. 20
    block : block size, e.g. 200x200x200
    roi   : e.g. nii_roi [0, 2176, 0, 232, 0, 2112]
    @return
    pad_index       : list of blocked ROI with padding around covering the large ROI `roi`,
                      such as [..., [[0, 220], [180, 232], [0, 220]], ...]
    effective_index : relative ROI for ODF_Index to pad_index,
                      such as [..., [[0, 200], [20, 52], [0, 200]], ...]
    ODF_Index       : list of blocked ROI with no overlap that tile the large ROI,
                      such as [..., [[0, 200], [200, 232], [0, 200]], ...]
    """
    pad = pad
    block = block
    shape = data.shape
    if roi is None:
        roi = [0, shape[2], 0, shape[0], 0, shape[1]]
    flag = [[i for i in range(roi[j*2], roi[j*2+1], block[j])]
            for j in range(3)]
    for i in range(len(flag)):
        flag[i].append(roi[i*2+1])
    pad_index = []
    effective_index = []
    ODF_Index = []
    for i in range(len(flag[0]) -1):
        for j in range(len(flag[1]) -1):
            for k in range(len(flag[2])-1 ):
                xl = max(flag[0][i] - pad, 0)
                xr = min(flag[0][i + 1] + pad, roi[1])
                yl = max(flag[1][j] - pad, 0)
                yr = min(flag[1][j + 1] + pad, roi[3])
                zl = max(flag[2][k] - pad, 0)
                zr = min(flag[2][k + 1] + pad, roi[5])
                ODF_Index.append(
                    [[flag[0][i], flag[0][i + 1]],
                     [flag[1][j], flag[1][j + 1]],
                     [flag[2][k], flag[2][k + 1]]])
                pad_index.append([[xl, xr], [yl, yr], [zl, zr]])
                effective_index.append([[flag[0][i] - xl, flag[0][i + 1] - xl],
                                        [flag[1][j] - yl, flag[1][j + 1] - yl],
                                        [flag[2][k] - zl, flag[2][k + 1] - zl]])
    return pad_index, effective_index, ODF_Index

def grad(temp):
    """compute grad of the image, with gaussian filter"""
    Vx = filter.gaussian_filter(temp, sigma * 1, order=[1, 0, 0],
                                mode='nearest', truncate=4)[::ratio, ::ratio,::ratio]
    Vy = filter.gaussian_filter(temp, sigma * 1, order=[0, 1, 0],
                                mode='nearest', truncate=4)[::ratio, ::ratio,::ratio]
    Vz = filter.gaussian_filter(temp, sigma * 1, order=[0, 0, 1],
                                mode='nearest', truncate=4)[::ratio, ::ratio,::ratio]
    return Vx, Vy, Vz

def psf(gradient, field_dirs, index):
    """
    For each gradient of the image, project it to predefined directions and adjust it 
    so that only the aligned direction has a large value.
        (r / ||r|| @ direction) ** index * ||r||
    Conceptually simulate the convolultion of the gradient with a sharp PSF.
    """
    gradient_norm = np.linalg.norm(gradient, axis=-1)[...,None]
    gradient = np.nan_to_num(gradient / gradient_norm)
    o = gradient @ field_dirs.T
    return o ** index * gradient_norm

sigma = 1
rho   = 20    # size of analyze step (also control analyze window and block size)
ratio = 2     # downsample ratio, i.e. step size for gradient computaion

def ComputeGradientDiffusionImage(img3d_path, dwi_path):
    # load 3D image
    h5 = h5py.File(img3d_path, 'r')
    img = h5['DataSet']['ResolutionLevel 2']['TimePoint 0']['Channel 1']['Data']
    ims_roi = [0, img.shape[0],
               0, img.shape[1],
               0, img.shape[2]]
    #ims_roi = [0, 200,
    #           500, 800,
    #           500, 800]
    nii_roi = [img.shape[2]-ims_roi[5], img.shape[2]-ims_roi[4],
               img.shape[0]-ims_roi[1], img.shape[0]-ims_roi[0],
               img.shape[1]-ims_roi[3], img.shape[1]-ims_roi[2]]
    print(f'Image ROI: {nii_roi}')
    ODF_Data = np.zeros([(nii_roi[1]-nii_roi[0]) // rho,
                         (nii_roi[3]-nii_roi[2]) // rho,
                         (nii_roi[5]-nii_roi[4]) // rho, 46], dtype=np.float32)
    print(f'Result image size: {ODF_Data.shape}')

    # directions of "gradient field". lmax=8 <=> 45 directions
    field_dirs = cp.loadtxt("./45.txt")[1:,:-1].astype(np.float32)

    padded_index, effective_index, ODF_Index = split(
        img, 1 * rho, [10 * rho, 10 *  rho, 10 * rho], nii_roi)
    dataSet = dataLoader.CustomIMSDataset(padded_index, effective_index, ODF_Index, img3d_path)
    dataloader = dataLoader.DataLoader(dataSet, batch_size=1, num_workers=10, pin_memory=True)
    ker1d = torch.tensor(cv2.getGaussianKernel(
            rho//ratio + 1, rho//ratio*10, cv2.CV_32F)[:, 0]).cuda(0)
    i = 0
    for batch in tqdm(dataloader, colour = 'GREEN'):
        # get blocked image
        roi = cp.asarray(batch[0][0])
        flag = cp.asarray(batch[1][0])[::ratio, ::ratio,::ratio]

        # get smoothed gradient
        Vx, Vy, Vz = grad(roi)
        Vx[flag] = 0
        Vy[flag] = 0
        Vz[flag] = 0
        r = cp.stack((Vx, Vy, Vz), axis=-1)
        # get simulated directional response
        x = psf(r, field_dirs, 100)
        # smooth the response and down sampling to resolution `rho`
        x = torch.as_tensor(x[None, ...].transpose([4, 0, 1, 2, 3])).cuda()
        x = F.conv3d(x, ker1d[None, None, None, None, :],
                     padding=(0, 0, ker1d.shape[0] // 2),
                     stride=[1, 1, rho // ratio])
        x = F.conv3d(x, ker1d[None, None, None, :, None],
                     padding=(0, ker1d.shape[0] // 2, 0),
                     stride=[1, rho // ratio, 1])
        x = F.conv3d(x, ker1d[None, None, :, None, None],
                     padding=(ker1d.shape[0] // 2, 0, 0),
                     stride=[rho // ratio, 1, 1])
        x = cp.array(x[..., None].transpose(0, -1)[0][0])
        cp.nan_to_num(x, 0)
        # add 0-th component
        x = cp.concatenate([cp.max(x[:], axis=-1)[..., None] * 2, x], axis=-1)
        x[..., 0] = cp.where(x[..., 0]>0.2, 1, 0)
        x = x.get()
        # write the tile (block) to result array
        odf = ODF_Index[i]
        eff = effective_index[i]
        ODF_Data[(odf[0][0]-nii_roi[0]) // rho : (odf[0][1]-nii_roi[0]) // rho,
                 (odf[1][0]-nii_roi[2]) // rho : (odf[1][1]-nii_roi[2]) // rho,
                 (odf[2][0]-nii_roi[4]) // rho : (odf[2][1]-nii_roi[4]) // rho] = \
            x[eff[0][0] // rho : eff[0][1] // rho,
              eff[1][0] // rho : eff[1][1] // rho,
              eff[2][0] // rho : eff[2][1] // rho]
        i += 1

    out = makeNiiForMRView(rho, ODF_Data)
    nib.save(out, dwi_path)

def makeNiiForMRView(fod_reslotion, out):
    out = nib.Nifti1Image(out, np.eye(4) * fod_reslotion*4/1000)
    out.affine[0, -1] = fod_reslotion / 2
    out.affine[1, -1] = fod_reslotion / 2
    out.affine[2, -1] = fod_reslotion / 2
    return out

def MRtrixProcess(dwi_path, n_tck_sample):
    res_path  = f'./response.nii'
    fod_path  = f'./FOD_sigma_{sigma}_rho_{rho}.nii'
    dir_path  = f'./45.txt'
    mask_path = f'mask.nii'
    tck_path  = f'tck.tck'

    # os.system(f'dwi2response tax {dwi}  {res}  -grad {dir_file}  -force')
    os.system(f'mrfilter "{dwi_path}" smooth -stdev 0.06,0.06,0.06 "{dwi_path}" -force')
    os.system(f'dwi2fod msmt_csd "{dwi_path}" "{res_path}" "{fod_path}" -grad "{dir_path}" -lmax 12 -shells 1000 -force')
    os.system(f'dwi2mask "{dwi_path}" "{mask_path}" -grad "{dir_path}" -force')
    os.system(f'tckgen -seed_image "{mask_path}" "{fod_path}" "{tck_path}" -select {n_tck_sample} -power 2 -cutoff 0.01 -force -minlength 0.01 -maxlength 100000 -force')
    os.system(f'mrview "{dwi_path}" -imagevisible false -tractography.load "{tck_path}"')

def ComputeCABLE(img3d_path, n_tck_sample):
    dwi_path = f'./DWI_sigma_{sigma}_rho_{rho}.nii'
    ComputeGradientDiffusionImage(img3d_path, dwi_path)
    MRtrixProcess(dwi_path, n_tck_sample)

def main():
    img3d_path = sys.argv[1]
    n_tck_sample = 10000
    ComputeCABLE(img3d_path, n_tck_sample)

if __name__ == '__main__':
    main()
