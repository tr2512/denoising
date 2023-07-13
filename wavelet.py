import scipy.stats
import numpy as np
from math import ceil
import pywt
import skimage.color as color
import numbers
import cv2

def _bayes_thresh(details, var):
    dvar = np.mean(details*details)
    return var / np.sqrt(max(dvar - var, 1e-6))

def sigma_estimation(detail_coeffs):
    detail_coeffs = detail_coeffs[detail_coeffs != 0]

    denom = scipy.stats.norm.ppf(0.75)
    sigma = np.median(np.abs(detail_coeffs)) / denom
    return sigma


def _wavelet_threshold(image, wavelet,
                       sigma=None, mode='soft', wavelet_levels=None):

    wavelet = pywt.Wavelet(wavelet)

    original_extent = tuple(slice(s) for s in image.shape)

    if wavelet_levels is None:
        dlen = wavelet.dec_len
        wavelet_levels = np.min(
            [pywt.dwt_max_level(s, dlen) for s in image.shape])

        wavelet_levels = max(wavelet_levels - 3, 1)

    coeffs = pywt.wavedecn(image, wavelet=wavelet, level=wavelet_levels)
    dcoeffs = coeffs[1:]
    if sigma is None:
        detail_coeffs = dcoeffs[-1]['d' * image.ndim]
        sigma = sigma_estimation(detail_coeffs)
    threshold = [{key: _bayes_thresh(level[key], sigma ** 2) for key in level}
                        for level in dcoeffs]

    denoised_detail = [{key: pywt.threshold(level[key],
                                            value=thresh[key],
                                            mode=mode) for key in level}
                                            for thresh, level in zip(threshold, dcoeffs)]
    denoised_coeffs = [coeffs[0]] + denoised_detail
    return pywt.waverecn(denoised_coeffs, wavelet)[original_extent]


def denoise_wavelet(image, sigma=None, wavelet='db1', mode='soft',
                    wavelet_levels=None, multichannel=False,
                    convert2ycbcr=False, method='BayesShrink',
                    ):
    if multichannel:
        if isinstance(sigma, numbers.Number) or sigma is None:
            sigma = [sigma] * image.shape[-1]

    if multichannel:
        out = color.rgb2ycbcr(image)
        for i in range(3):
            # renormalizing this color channel to live in [0, 1]
            min, max = out[..., i].min(), out[..., i].max()
            channel = out[..., i] - min
            channel /= max - min
            out[..., i] = denoise_wavelet(channel, wavelet=wavelet,
                                          sigma=sigma[i],
                                          mode=mode,
                                          wavelet_levels=wavelet_levels)

            out[..., i] = out[..., i] * (max - min)
            out[..., i] += min
        out = color.ycbcr2rgb(out)
    else:
        out = _wavelet_threshold(image, wavelet=wavelet,
                                 sigma=sigma, mode=mode,
                                 wavelet_levels=wavelet_levels)

    return np.clip(out, 0, 1)

def inference_wavelet(img_dir):
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    denoised = denoise_wavelet(img, sigma=0.1, multichannel=True, convert2ycbcr=True)
    return (denoised * 255).astype(np.uint8)
