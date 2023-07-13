import math
import numpy as np
import cv2


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def cut_image(img, size):
    img = img[:, :, :3]
    h, w, _ = img.shape
    nrow = (h - 1) // size[0] + 1
    ncol = (w - 1) // size[1] + 1
    hcut = size[0]
    wcut = size[1]
    cuts = []
    indexes = []
    for i in range(nrow):
        for j in range(ncol):
            if (i + 1) * hcut <= h and (j + 1) * wcut <= w:
                cuts.append(img[i*hcut:(i + 1)*hcut, j*wcut:(j + 1)*wcut, :])
            elif (i + 1) * hcut <= h and (j + 1) * wcut > w:
                cuts.append(img[i*hcut:(i + 1)*hcut, j*wcut:w, :])
            elif (i + 1) * hcut > h and (j + 1) * wcut <= w:
                cuts.append(img[i*hcut:h, j*wcut:(j + 1)*wcut, :])
            else:
                cuts.append(img[i*hcut:h, j*wcut:w, :])
            indexes.append((i, j))
    return cuts, indexes

def recover_image(imgs, indexes, original_size, crop_size):
    h, w = original_size
    recover = np.zeros((h, w, 3))
    hcrop, wcrop = crop_size
    for index, img in zip(indexes, imgs):
        i, j = index
        if (i + 1) * hcrop <= h and (j + 1) * wcrop <= w:
            recover[i*hcrop:(i + 1)*hcrop, j*wcrop: (j + 1)*wcrop, :] = img
        elif (i + 1) * hcrop <= h and (j + 1) * wcrop > w:
            recover[i*hcrop:(i + 1)*hcrop, j*wcrop:w, :] = img
        elif (i + 1) * hcrop > h and (j + 1) * wcrop <= w:
            recover[i*hcrop:h, j*wcrop:(j + 1) * wcrop, :] = img
        else:
            recover[i*hcrop:h, j*wcrop:w, :] = img
    return recover