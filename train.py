import torch
import torch.nn as nn
import torch.nn.functional as F
from logger import setup_logger
from denoise_model import Denoise
import argparse
from dataset import SIDDDataset
from utils import calculate_psnr, calculate_ssim


class ReconstructionLoss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-3):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == 'l2':
            return torch.mean(torch.sum((x - target)**2, (1, 2, 3)))
        elif self.losstype == 'l1':
            diff = x - target
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3)))
        elif self.losstype == 'l_log':
            diff = x - target
            eps = 1e-6
            return torch.mean(torch.sum(-torch.log(1-diff.abs()+eps), (1, 2, 3)))
        else:
            print("reconstruction loss type error!")
            return 0

loss_forward = ReconstructionLoss('l2')
loss_backward = ReconstructionLoss('l1')
parser = argparse.ArgumentParser()
parser.add_argument('-imgtraindir', type=str)
parser.add_argument('--lbltraindir', type=str)
parser.add_argument('--imgvaldir', type=str)
parser.add_argument('--lblvaldir', type=str)
args = parser.parse_args()

logger = setup_logger("InvDN", ".", "train.log")
logger.info("Start training")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Denoise()
model.to(device)

optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-4,
                                                weight_decay=1e-8,
                                                betas=(0.9, 0.999))

optimizer.zero_grad()
traindataset = SIDDDataset(args.imgtraindir, args.lbltraindir, 'train')
valdataset = SIDDDataset(args.imgvaldir, args.lblvaldir, 'val')

trainloader = torch.utils.data.DataLoader(traindataset, batch_size=28, shuffle=True, drop_last=True, pin_memory=True)
valloader = torch.utils.data.DataLoader(valdataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True)

model.train()
iteration = 0
while iteration < 600000:
    for i, (images, labels, lrs) in enumerate(trainloader):
        if iteration % 100000 == 0 and iteration > 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 2
        model.train()
        optimizer.zero_grad()
        images = images.to(device).float()
        labels = labels.to(device)
        lrs = lrs.to(device)

        output_lrs, output_denoises = model(images)
        loss = 16 * loss_forward(output_lrs, lrs) + loss_backward(output_denoises, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.net.parameters(), 10)
        optimizer.step()
        logger.info(f'Training loss: {loss}')
        iteration += 1
        if iteration % 200 == 0:
            model.eval()
            logger.info("Eval mode")
            psnr = []
            ssim = []
            for images, labels in valloader:
                images = images.to(device).float()
                with torch.no_grad():
                    outputs = model(images)
                outputs = outputs.squeeze().transpose((1, 2, 0))
                psnr.append(calculate_psnr(outputs*255, labels.squeeze().permute(1, 2, 0).numpy()*255))
                ssim.append(calculate_ssim(outputs*255, labels.squeeze().permute(1, 2, 0).numpy()*255*255))
            logger.info(f'Average psnr is: {sum(psnr)/ len(psnr)}')
            logger.info(f'Average ssim is: {sum(ssim)/ len(ssim)}')
            torch.save(model.net.state_dict(), "model.pkl")
