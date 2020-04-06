import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def ssim_value(original_img, other_img):
    return ssim(original_img, other_img,
                data_range=np.max(other_img) - np.min(other_img),
                multichannel=True)


def psnr_value(original_img, other_img):
    return psnr(original_img, other_img,
                data_range=np.max(other_img) - np.min(other_img))


def format(value):
    return "%.2f" % value
