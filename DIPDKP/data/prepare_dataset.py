# Generate random Gaussian kernels and downscale images
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import center_of_mass, shift
import glob
from scipy.io import savemat
import os
from PIL import Image
import torch
import torch.nn.functional as F
import argparse


# Function for centering a kernel
def kernel_shift(kernel, sf):
    current_center_of_mass = center_of_mass(kernel)
    wanted_center_of_mass = (np.array(kernel.shape) - sf) / 2.
    shift_vec = wanted_center_of_mass - current_center_of_mass
    return shift(kernel, shift_vec)


def analytic_kernel(k):
    k_size = k.shape[0]
    big_k = np.zeros((3 * k_size - 2, 3 * k_size - 2))
    for r in range(k_size):
        for c in range(k_size):
            big_k[2 * r:2 * r + k_size, 2 * c:2 * c + k_size] += k[r, c] * k
    crop = k_size // 2
    cropped_big_k = big_k[crop:-crop, crop:-crop]
    return cropped_big_k / cropped_big_k.sum()


def gen_kernel_fixed(k_size, scale_factor, lambda_1, lambda_2, theta, noise):
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    MU = k_size // 2 + 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]

    [X, Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    ZZ = Z - MU
    ZZ_t = ZZ.transpose(0, 1, 3, 2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)
    raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)
    kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
    return kernel


def gen_kernel_random(k_size, scale_factor, min_var, max_var, noise_level):
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2
    kernel = gen_kernel_fixed(k_size, scale_factor, lambda_1, lambda_2, theta, noise)
    return kernel


def degradation(input, kernel, scale_factor, noise_im, device=torch.device('cuda')):
    input = torch.from_numpy(input).type(torch.FloatTensor).to(device).unsqueeze(0).permute(3, 0, 1, 2)
    input = F.pad(input, pad=(kernel.shape[0] // 2, kernel.shape[0] // 2,
                              kernel.shape[0] // 2, kernel.shape[0] // 2), mode='circular')
    kernel = torch.from_numpy(kernel).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)

    output = F.conv2d(input, kernel)
    output = output.permute(2, 3, 0, 1).squeeze(3).cpu().numpy()
    output = output[::scale_factor[0], ::scale_factor[1], :]
    output += np.random.normal(0, np.random.uniform(0, noise_im), output.shape)
    return output


def degradation_regionwise(input, kernels, scale_factor, noise_im, grid_size=2, device=torch.device('cuda')):
    """
    Apply different kernels on different grid regions.
    input: 原始图像 (H, W, C)
    kernels: list of kernels, 数量应为 grid_size*grid_size
    grid_size: 网格大小 (e.g. 2 -> 2x2 网格)
    """
    H, W, C = input.shape
    h_step, w_step = H // grid_size, W // grid_size

    outputs = []
    for i in range(grid_size):
        row_patches = []
        for j in range(grid_size):
            x0, x1 = j * w_step, (j + 1) * w_step if j < grid_size - 1 else W
            y0, y1 = i * h_step, (i + 1) * h_step if i < grid_size - 1 else H
            patch = input[y0:y1, x0:x1, :]

            kernel = kernels[i * grid_size + j]
            patch_t = torch.from_numpy(patch).type(torch.FloatTensor).to(device).unsqueeze(0).permute(3, 0, 1, 2)
            patch_t = F.pad(patch_t, pad=(kernel.shape[0] // 2,) * 4, mode='circular')
            kernel_t = torch.from_numpy(kernel).type(torch.FloatTensor).to(device).unsqueeze(0).unsqueeze(0)

            out = F.conv2d(patch_t, kernel_t)
            out = out.permute(2, 3, 0, 1).squeeze(3).cpu().numpy()
            out = out[::scale_factor[0], ::scale_factor[1], :]
            row_patches.append(out)
        outputs.append(np.concatenate(row_patches, axis=1))

    output = np.concatenate(outputs, axis=0)
    output += np.random.normal(0, np.random.uniform(0, noise_im), output.shape)
    return output


def modcrop(img_in, scale):
    img = np.copy(img_in)
    if img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


def generate_dataset(images_path, out_path_im, out_path_ker, k_size, scale_factor,
                     min_var, max_var, noise_ker, noise_im, kernelgan_x4=False, regionwise=False, grid_size=2):
    os.makedirs(out_path_im, exist_ok=True)
    os.makedirs(out_path_ker, exist_ok=True)

    files_source = glob.glob(images_path)
    files_source.sort()
    for i, path in enumerate(files_source):
        print(path)
        im = np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.
        im = modcrop(im, scale_factor[0])

        if regionwise:
            num_regions = grid_size * grid_size
            kernels = [gen_kernel_random(k_size, scale_factor, min_var, max_var, noise_ker) for _ in range(num_regions)]
            lr = degradation_regionwise(im, kernels, scale_factor, noise_im,
                                        grid_size=grid_size,
                                        device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            for idx, ker in enumerate(kernels):
                savemat('%s/%s_part%d.mat' % (out_path_ker, os.path.splitext(os.path.basename(path))[0], idx),
                        {'Kernel': ker})
                plot_kernel(ker, '%s/%s_part%d.png' % (out_path_ker, os.path.splitext(os.path.basename(path))[0], idx))
        else:
            if kernelgan_x4:
                kernel = gen_kernel_random(k_size, 2, min_var, max_var, noise_ker)
                kernel = analytic_kernel(kernel)
                kernel = kernel_shift(kernel, 4)
            else:
                kernel = gen_kernel_random(k_size, scale_factor, min_var, max_var, noise_ker)

            lr = degradation(im, kernel, scale_factor, noise_im,
                             device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

            savemat('%s/%s.mat' % (out_path_ker, os.path.splitext(os.path.basename(path))[0]), {'Kernel': kernel})
            plot_kernel(kernel, '%s/%s.png' % (out_path_ker, os.path.splitext(os.path.basename(path))[0]))

        plt.imsave('%s/%s.png' % (out_path_im, os.path.splitext(os.path.basename(path))[0]),
                   np.clip(lr, 0, 1), vmin=0, vmax=1)


def plot_kernel(gt_k_np, savepath):
    plt.clf()
    f, ax = plt.subplots(1, 1, figsize=(6, 4), squeeze=False)
    im = ax[0, 0].imshow(gt_k_np, vmin=0, vmax=gt_k_np.max())
    plt.colorbar(im, ax=ax[0, 0])
    plt.savefig(savepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DIPDKP',
                        help='DIPDKP, generate data blurred by anisotropic Gaussian kernel.')
    parser.add_argument('--sf', type=int, default=4, help='scale factor: 2, 3, 4, 8')
    parser.add_argument('--dataset', type=str, default='Set5',
                        help='dataset: Set5, Set14, BSD100, Urban100, DIV2K')
    parser.add_argument('--noise_ker', type=float, default=0, help='noise on kernel')
    parser.add_argument('--noise_im', type=float, default=0, help='noise on LR image')
    parser.add_argument('--regionwise', action='store_true',
                        help='Apply different blur kernels to grid regions of the image')
    parser.add_argument('--grid_size', type=int, default=2,
                        help='Grid size for regionwise blur (default=2 means 2x2 grid)')
    opt = parser.parse_args()

    work_path = os.path.dirname(__file__)

    images_path = '{}/datasets/{}/HR/*.png'.format(work_path, opt.dataset)
    out_path_im = '{}/datasets/{}/{}_lr_x{}'.format(work_path, opt.dataset, opt.model, opt.sf)
    out_path_ker = '{}/datasets/{}/{}_gt_k_x{}'.format(work_path, opt.dataset, opt.model, opt.sf)

    if opt.model == 'DIPDKP':
        min_var = 0.175 * opt.sf
        max_var = min(2.5 * opt.sf, 10)
        k_size = np.array([min(opt.sf * 4 + 3, 21), min(opt.sf * 4 + 3, 21)])
        generate_dataset(images_path, out_path_im, out_path_ker, k_size, np.array([opt.sf, opt.sf]),
                         min_var, max_var, opt.noise_ker, opt.noise_im,
                         regionwise=opt.regionwise, grid_size=opt.grid_size)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    main()
    sys.exit()
