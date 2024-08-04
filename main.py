import torch
import h5py
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import dataLoader
import cupyx.scipy.ndimage as filter
import cupy as cp
import os
import sys
from tqdm import tqdm
dirs = cp.loadtxt("./45.txt")[1:,:-1].astype(np.float32)
sigma = 1
rho = 20
ratio = 2
dwi = f'./DWI_sigma_{sigma}_rho_{rho}.nii'
dwi_denoise = f'./DWI_sigma_{sigma}_rho_{rho}_denoise.nii'
res = f'./response.nii'
fod = f'./FOD_sigma_{sigma}_rho_{rho}.nii'
dir_file = f'./45.txt'


def partition(block_size, shape, unit):
    flags = [
        [i for i in range(shape[2 * j], shape[2 * j + 1] + 1, block_size[j])]
        for j in range(3)
    ]
    for i in range(3):
        if flags[i][-1] != shape[2 * i + 1]:
            flags[i].append(shape[2 * i + 1] - (shape[2 * i + 1] % unit))
    ODF_partition = []
    for i in range(len(flags[0]) - 1):
        for j in range(len(flags[1]) - 1):
            for k in range(len(flags[2]) - 1):
                ODF_partition.append([
                    [flags[0][i], flags[0][i + 1]],
                    [flags[1][j], flags[1][j + 1]],
                    [flags[2][k], flags[2][k + 1]]
                ])
    return ODF_partition


def grad(temp):
    Vx = filter.gaussian_filter(temp, sigma * 1, order=[1, 0, 0])[::ratio, ::ratio, ::ratio]
    Vy = filter.gaussian_filter(temp, sigma * 1, order=[0, 1, 0])[::ratio, ::ratio, ::ratio]
    Vz = filter.gaussian_filter(temp, sigma * 1, order=[0, 0, 1])[::ratio, ::ratio, ::ratio]
    return Vx, Vy, Vz
def psf(gradient, index):
    gradient_norm = np.linalg.norm(gradient, axis=-1)[...,None]
    gradient = np.nan_to_num(gradient / gradient_norm)
    o = gradient @ dirs.T
    return o ** index * gradient_norm
def makeNiiForMRView(fod_reslotion, out):
    out = nib.Nifti1Image(out, np.eye(4) * fod_reslotion*4/1000)
    return out
def main():
    address = sys.argv[1]
    h5 = h5py.File(         address,        'r')
    img = h5['DataSet']['ResolutionLevel 2']['TimePoint 0']['Channel 1']['Data']
    ims_roi = [0,img.shape[0],
               0,img.shape[1],
               0,img.shape[2]]
    nii_roi = [img.shape[2]-ims_roi[5], img.shape[2]-ims_roi[4],
               img.shape[0]-ims_roi[1] , img.shape[0]-ims_roi[0],
               img.shape[1]-ims_roi[3], img.shape[1]-ims_roi[2]]
    ODF_out = np.zeros([(nii_roi[1]-nii_roi[0]) // rho,
                      (nii_roi[3]-nii_roi[2]) // rho,
                      (nii_roi[5]-nii_roi[4]) // rho, 46], dtype=np.float32)
    ODF_partition = partition([10 * rho, 10 * rho, 10 * rho], nii_roi, rho)
    dataSet = dataLoader.CustomIMSDataset(ODF_partition,address)
    dataloader = dataLoader.DataLoader(dataSet,batch_size=1,num_workers=11,pin_memory=True)
    k = torch.ones(rho//ratio).cuda(0)/(rho//ratio)
    for batch in tqdm(dataloader, colour = 'GREEN'):
        block = cp.asarray(batch[0][0])
        Vx, Vy, Vz = grad(block)
        d = cp.stack(( Vx, Vy, Vz), axis=-1)
        x = psf(d,100)
        x = torch.as_tensor(x[None, ...].transpose([4, 0, 1, 2, 3])).cuda()
        x = F.conv3d(x, k[None, None, None, None, :], stride=[1, 1, rho // ratio])
        x = F.conv3d(x, k[None, None, None, :, None], stride=[1, rho // ratio, 1])
        x = F.conv3d(x, k[None, None, :, None, None], stride=[rho // ratio, 1, 1])
        x = cp.array(x[..., None].transpose(0, -1)[0][0])
        cp.nan_to_num(x, 0)
        x = cp.concatenate([cp.max(x[:], axis=-1)[..., None] * 2, x], axis=-1)
        x[...,0] = cp.where(x[...,0]>0.2,1,0)
        x = x.get()
        ODF_block = batch[1]
        ODF_out[ (ODF_block[0][0]-nii_roi[0]) // rho:(ODF_block[0][1]-nii_roi[0]) // rho,(ODF_block[1][0]-nii_roi[2]) // rho:(ODF_block[1][1]-nii_roi[2]) // rho,(ODF_block[2][0]-nii_roi[4]) // rho:(ODF_block[2][1]-nii_roi[4]) // rho] = x
    out = makeNiiForMRView(rho, ODF_out)
    nib.save(out, dwi)

    os.system(f'mrfilter  {dwi} smooth -stdev 0.04,0.04,0.04 {dwi}   -force')
    os.system(f'dwi2fod msmt_csd {dwi} {res} {fod} -grad {dir_file} -lmax 8 -shells 1000 -force')
    os.system(f'dwi2mask {dwi} mask.nii -grad {dir_file} -force')
    os.system(f'tckgen -seed_image mask.nii {fod} tck.tck -select 20000 -power 2 -cutoff 0.05 -force -minlength 0.1 -maxlength 100000 -force')
    os.system(f'mrview  {dwi} -imagevisible false -tractography.load tck.tck -tractography.opacity 0.5')

if __name__ == '__main__':
    main()