# If you find this code useful in your research, consider citing the following paper:
# Yue Zhang et al. Whole-brain reconstruction of fiber tracts based on the 3-D cytoarchitectonic organization
import os
import sys
import utils.dataLoader as dataLoader
import numpy as np
import h5py
import nibabel as nib

import torch
import torch.nn.functional as F
from scipy import ndimage as filter  # Use CPU-based Gaussian filtering
from tqdm import tqdm

def split(block_size, shape, unit):
    """
    Split the image into blocks.
    
    @param block_size: block size, e.g., [200, 200, 200]
    @param shape: image shape boundaries, e.g., [start0, end0, start1, end1, start2, end2]
    @param unit: resolution unit, e.g., 20
    @return: A list of blocked ROIs, e.g., [..., [[0, 200], [0, 200], [0, 200]], ...]
    """
    flags = [
        [i for i in range(shape[2 * j], shape[2 * j + 1] + 1, block_size[j])]
        for j in range(3)
    ]
    for i in range(3):
        if flags[i][-1] != shape[2 * i + 1] - ((shape[2 * i + 1] - shape[2 * i]) % unit):
            flags[i].append(shape[2 * i + 1] - ((shape[2 * i + 1] - shape[2 * i]) % unit))
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

def grad(temp, sigma, ratio):
    """
    Compute the gradient of the image using Gaussian smoothing.
    
    @param temp: input image data
    @param sigma: standard deviation for Gaussian filter
    @param ratio: downsampling factor
    @return: gradients along x, y, and z directions
    """
    Vx = filter.gaussian_filter(temp, sigma * 1, order=[1, 0, 0],
                                mode='nearest', truncate=4)[::ratio, ::ratio, ::ratio]
    Vy = filter.gaussian_filter(temp, sigma * 1, order=[0, 1, 0],
                                mode='nearest', truncate=4)[::ratio, ::ratio, ::ratio]
    Vz = filter.gaussian_filter(temp, sigma * 1, order=[0, 0, 1],
                                mode='nearest', truncate=4)[::ratio, ::ratio, ::ratio]
    return Vx, Vy, Vz

def psf(gradient, field_dirs, index):
    """
    For each gradient of the image, project it to predefined directions and adjust it
    so that only the aligned direction has a large value.
    
    Calculation: (r / ||r|| @ direction) ** index * ||r||
    This simulates the convolution of the gradient with a sharp point spread function (PSF).
    
    @param gradient: image gradient vectors
    @param field_dirs: predefined gradient directions
    @param index: exponent to sharpen the response
    @return: modified gradient response
    """
    gradient_norm = np.linalg.norm(gradient, axis=-1)[..., None]
    # Avoid division by zero
    gradient = np.nan_to_num(gradient / gradient_norm)
    x = gradient @ field_dirs.T
    return x ** index * gradient_norm

def GradientWeightedFunction(img3d_path, cable_params, dwi_path):
    sigma = cable_params['sigma']
    rho = cable_params['rho']
    ratio = cable_params['ratio']
    dirs = cable_params['dir_path']
    
    # Load the 3D image from HDF5 file
    h5_file = h5py.File(img3d_path, 'r')
    img = h5_file['DataSet']['ResolutionLevel 0']['TimePoint 0']['Channel 0']['Data']
    ims_roi = [0, img.shape[0],
               0, img.shape[1],
               0, img.shape[2]]
    # Calculate ROI boundaries for NIfTI format (note the order change)
    nii_roi = [img.shape[2] - ims_roi[5], img.shape[2] - ims_roi[4],
               img.shape[0] - ims_roi[1], img.shape[0] - ims_roi[0],
               img.shape[1] - ims_roi[3], img.shape[1] - ims_roi[2]]
    print(f'Image ROI: {nii_roi}')
    
    # Initialize result array for Gradient Weighted Function (GWF)
    ODF_Data = np.zeros([(nii_roi[1] - nii_roi[0]) // rho,
                         (nii_roi[3] - nii_roi[2]) // rho,
                         (nii_roi[5] - nii_roi[4]) // rho, 46], dtype=np.float32)
    print(f'Result GWF image size: {ODF_Data.shape}')
    
    # Load the predefined gradient directions (starting from second row)
    field_dirs = np.loadtxt(dirs)[1:, :-1].astype(np.float32)
    ODF_partition = split([10 * rho, 10 * rho, 10 * rho], nii_roi, rho)
    
    # Create the dataset and dataloader for processing image blocks
    dataSet = dataLoader.CustomIMSDataset(ODF_partition, img3d_path)
    dataloader = dataLoader.DataLoader(dataSet, batch_size=1, num_workers=0, pin_memory=False)
    
    # Construct a 1D averaging kernel
    ker1d = torch.ones(rho // ratio) / (rho // ratio)

    for batch in tqdm(dataloader, colour='GREEN'):
        # Get the image block and convert to numpy array
        roi = np.asarray(batch[0][0])
        # Compute the smoothed gradients
        Vx, Vy, Vz = grad(roi, sigma, ratio)
        r = np.stack((Vx, Vy, Vz), axis=-1)
        # Compute the simulated directional response using PSF
        x = psf(r, field_dirs, 100)
        # Convert to torch tensor and adjust dimensions for 3D convolution
        x = torch.as_tensor(x[None, ...].transpose([4, 0, 1, 2, 3]))
        x = F.conv3d(x, ker1d[None, None, None, None, :], stride=[1, 1, rho // ratio])
        x = F.conv3d(x, ker1d[None, None, None, :, None], stride=[1, rho // ratio, 1])
        x = F.conv3d(x, ker1d[None, None, :, None, None], stride=[rho // ratio, 1, 1])
        # Rearrange dimensions and convert tensor to numpy array
        x = x[..., None].transpose(0, -1)[0][0]
        x = x.numpy()
        np.nan_to_num(x, copy=False)
        # Add the 0-th component by concatenating the max response multiplied by 2
        x = np.concatenate([np.max(x, axis=-1)[..., None] * 2, x], axis=-1)
        x[..., 0] = np.where(x[..., 0] > 0.2, 1, 0)
        # Write the processed block into the result array
        ODF_block = batch[1]
        ODF_Data[(ODF_block[0][0] - nii_roi[0]) // rho:(ODF_block[0][1] - nii_roi[0]) // rho,
                 (ODF_block[1][0] - nii_roi[2]) // rho:(ODF_block[1][1] - nii_roi[2]) // rho,
                 (ODF_block[2][0] - nii_roi[4]) // rho:(ODF_block[2][1] - nii_roi[4]) // rho] = x

    out = makeNiiForMRView(rho, ODF_Data)
    nib.save(out, dwi_path)

def makeNiiForMRView(fod_resolution, out):
    """
    Create a NIfTI image from the processed data.
    
    @param fod_resolution: resolution factor
    @param out: output image data
    @return: NIfTI image for visualization
    """
    nii_img = nib.Nifti1Image(out, np.eye(4) * fod_resolution * 4 / 1000)
    nii_img.affine[0, -1] = fod_resolution / 2
    nii_img.affine[1, -1] = fod_resolution / 2
    nii_img.affine[2, -1] = fod_resolution / 2
    return nii_img

def ODFProcessing(dwi_path, cable_params, fod_path, mask_path, tck_path):
    """
    Perform ODF processing by executing external commands.
    
    @param dwi_path: path to the DWI image
    @param cable_params: parameter dictionary
    @param fod_path: path for the FOD output
    @param mask_path: path for the mask output
    @param tck_path: path for the tractography output
    """
    n_tck_sample = cable_params['n_tck_sample']
    res_path     = cable_params['response_path']
    dir_path     = cable_params['dir_path']

    # Generate smoothed DWI image, FOD, mask, and tractography using external commands
    # os.system(f'dwi2response tax {dwi_path} {res_path} -grad {dir_path} -force')
    os.system(f'mrfilter "{dwi_path}" smooth -stdev 0.04,0.04,0.04 "{dwi_path}" -force')
    os.system(f'dwi2fod msmt_csd "{dwi_path}" "{res_path}" "{fod_path}" -grad "{dir_path}" '
              f'-lmax 12 -shells 1000 -force')
    os.system(f'dwi2mask "{dwi_path}" "{mask_path}" -grad "{dir_path}" -force')
    os.system(f'tckgen -seed_image "{mask_path}" "{fod_path}" "{tck_path}" -select {n_tck_sample} '
              f'-power 2 -cutoff 0.1 -force -minlength 0.01 -maxlength 100000 -force')

def ComputeCABLE(img3d_path, cable_params):
    """
    Compute the CABLE pipeline.
    
    @param img3d_path: path to the input 3D image
    @param cable_params: parameter dictionary for processing
    @return: dictionary containing paths for DWI, FOD, tractography, and mask
    """
    sigma = cable_params['sigma']
    rho   = cable_params['rho']
    dwi_path  = f'./DWI_sigma_{sigma}_rho_{rho}.nii'
    fod_path  = f'./FOD_sigma_{sigma}_rho_{rho}.nii'
    tck_path  = f'tck.tck'
    mask_path = f'mask.nii'
    GradientWeightedFunction(img3d_path, cable_params, dwi_path)
    ODFProcessing(dwi_path, cable_params, fod_path, mask_path, tck_path)
    res_path_set = {
        'dwi_path': dwi_path,
        'fod_path': fod_path,
        'tck_path': tck_path,
        'mask_path': mask_path,
    }
    return res_path_set

def main():
    if len(sys.argv) < 2:
        sys.exit("Please provide the 3D image file path as an argument.")
    img3d_path = sys.argv[1]
    cable_params = {
        # Smoothing parameter for gradient computation
        'sigma': 1,
        # Step size for analysis (controls analysis window and block size)
        'rho'  : 20,
        # Downsampling ratio, i.e., step size for gradient computation
        'ratio': 2,
        'n_tck_sample': 10000,
        'dir_path': './utils/45.txt',
        'response_path': './utils/response.nii',
    }
    res_path_set = ComputeCABLE(img3d_path, cable_params)
    # Visualize the results using an external MR viewer
    dwi_path = res_path_set['dwi_path']
    tck_path = res_path_set['tck_path']
    os.system(f'mrview "{dwi_path}" -imagevisible false -tractography.opacity 0.35 -tractography.load "{tck_path}"')

if __name__ == '__main__':
    main()
