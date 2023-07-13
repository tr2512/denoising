import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from invertnet import InvNet
import cv2
from utils import cut_image, recover_image

class Denoise(nn.Module):

    def __init__(self):
        super(Denoise, self).__init__()
        self.net = InvNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_test(x)

    def forward_train(self, x):
        output_forward = self.net(x, reverse=False)
        output_lr = output_forward[:, :3, :, :]
        gaussian_noise = torch.randn(output_forward[:, 3:, :, :].shape, device=self.device)

        x_backward = torch.cat([output_lr, gaussian_noise], dim=1)
        y_backward = self.net(x_backward, reverse=True)
        output_denoise = y_backward[:, :3, :, :]

        return output_lr, output_denoise

    def forward_test(self, x):
        x = x.cpu().numpy()
        xs = [x]
        xs.append(x[:, :, :, ::-1].copy())
        xs.append(x[:, :, ::-1, :].copy())
        xs.append(x.transpose((0, 1, 3, 2)).copy())
        output_forward = [self.net(torch.tensor(aug, device=self.device), reverse=False) for aug in xs]
        x_backwards = []
        for data in output_forward:
            output_lr = data[:, :3, :, :]
            gaussian_noise = torch.randn(data[:, 3:, :, :].shape, device=self.device)
            x_backward = torch.cat([output_lr, gaussian_noise], dim=1)
            x_backwards.append(x_backward)

        output_denoise = [self.net(data, reverse=True) for data in x_backwards]
        output_denoise[0] = np.array(output_denoise[0].cpu())
        output_denoise[1] = np.array(output_denoise[1].cpu())[:, :, :, ::-1].copy()
        output_denoise[2] = np.array(output_denoise[2].cpu())[:, :, ::-1, :].copy()
        output_denoise[3] = np.array(output_denoise[3].cpu()).transpose((0, 1, 3, 2)).copy()
  
        output_cat = np.concatenate(output_denoise, axis=0)
        output = output_cat.mean(axis=0, keepdims=True)

        return output

def inference_invdn(img_dir):
    model = Denoise()
    model.eval()
    model.net.load_state_dict(torch.load("invdn_checkpoint.pth", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    image = cv2.imread(img_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255
    cuts, indexes = cut_image(image, (256, 256))
    h, w = image.shape[:2]
    outputs = []
    with torch.no_grad():
        for cut in cuts:
            h1, w1 = cut.shape[:2]
            cut = cv2.resize(cut, (256, 256), interpolation=cv2.INTER_LINEAR)
            cut = cut.transpose(2, 0, 1)
            output = model(torch.tensor(cut).unsqueeze(0).float())
            output = output[0].transpose(1, 2, 0) * 255
            output = output.astype(dtype=np.uint8)
            output = cv2.resize(output, (w1, h1), interpolation=cv2.INTER_LINEAR)
            outputs.append(output)
    full = recover_image(outputs, indexes, (h, w), (256, 256))
    return full.astype(np.uint8)

