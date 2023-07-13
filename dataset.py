import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from glob import glob
import numpy as np
import random


def transform(imgs):
    if random.random() < 0.5:
        imgs = [img[::-1, :, :].copy() for img in imgs]
    if random.random() < 0.5:
        imgs = [img[:, ::-1, :].copy() for img in imgs]
    if random.random() < 0.5:
        imgs = [img.transpose((1, 0, 2)).copy() for img in imgs]
    return imgs


class SIDDDataset(torch.utils.data.Dataset):

    def __init__(self, img_dir, lbl_dir, mode='train'):
        super(SIDDDataset, self).__init__()
        self.img_root = img_dir
        self.lbl_root = lbl_dir
        self.img_list = glob(img_dir + "/*")
        self.lbl_list = glob(lbl_dir + "/*")
        self.mode = mode
        self.img_size = 144
        print(len(self.img_list))
        print(len(self.lbl_list))
        assert len(self.img_list) == len(self.lbl_list)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_dir = self.img_list[idx]
        lbl_dir = self.img_list[idx]
        img = cv2.imread(img_dir)
        lbl = cv2.imread(lbl_dir)
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255).astype(np.float64)
        lbl = (cv2.cvtColor(lbl, cv2.COLOR_BGR2RGB) / 255).astype(np.float64)
        lr = cv2.resize(lbl, dsize=(img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_CUBIC)
        if self.mode == 'train':
            H, W, _ = lr.shape
            lr_size = self.img_size // 4
            h = random.randint(0, H - lr_size)
            w = random.randint(0, W - lr_size)
            lr = lr[h:h + lr_size, w:w + lr_size, :]
            h4 = h * 4
            w4 = w * 4
            img = img[h4:h4 + self.img_size, w4:w4 + self.img_size, :]
            lbl = lbl[h4:h4 + self.img_size, w4:w4 + self.img_size, :]
            img, lbl, lr = transform([img, lbl, lr])
            return torch.tensor(img).permute(2, 0, 1), torch.tensor(lbl).permute(2, 0, 1), torch.tensor(lr).permute(2, 0, 1)
        else:
            return torch.tensor(img[:self.img_size, :self.img_size, :]).permute(2, 0, 1), torch.tensor(lbl[:self.img_size, :self.img_size, :]).permute(2, 0, 1)


