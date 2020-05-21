# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import time
import pandas as pd
import easydict
import os
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torch.nn.functional as F
from torchvision import models, transforms
# -

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

args = easydict.EasyDict({
        "train_root_dir": "../data/main/train",
        "val_root_dir": "../data/main/val",
        "batch_size": 4,
        "size": (480, 640),
        "num_worker": 1,
        "lr": 1e-4,
        "start_epoch": 1,
        "result_path": './results_AODNet',
})

device_str = "cuda:3"
device = torch.device(device_str if torch.cuda.is_available() else "cpu")


class HazeDataset(data.Dataset):
    def __init__(self,  root_dir, size=(480, 640)):
        self.root_dir, _, self.files = next(os.walk(root_dir))
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = os.path.join(self.root_dir, self.files[idx])

        data = np.load(name, allow_pickle=True)

        hazy     = torch.FloatTensor(data[0])
        hazefree = torch.FloatTensor(data[1])
        
        return hazy, hazefree


# +
train_dataset = HazeDataset(args.train_root_dir, args.size)
val_dataset = HazeDataset(args.val_root_dir, args.size)

# 動作確認
index = 0
print(train_dataset.__getitem__(index)[0].size())
print(train_dataset.__getitem__(index)[1].size())

# +
train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# 動作確認
batch_iterator = iter(dataloaders_dict["train"])
hazies, hazefrees = next(batch_iterator)
print(hazies.size())
print(hazefrees.size())
print("trainのバッチ数 : ", len(dataloaders_dict["train"]))


# -

class AODnet(nn.Module):   
    def __init__(self):
        super(AODnet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=3, padding=1)
        self.b = 1

    def forward(self, x):  
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3),1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4),1)
        k = F.relu(self.conv5(cat3))

        if k.size() != x.size():
            raise Exception("k, haze image are different size!")

        output = k * x - k + self.b
        return F.relu(output)


def generate_model():
    net = AODnet()
    net.eval()
    return net


# +
net = generate_model()
print("使用デバイス:", device)

print('ネットワーク設定完了')
# -

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=53760, gamma=0.5)


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    
    net.to(device)
    torch.backends.cudnn.benchmark = True
    
    iteration = 1
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    logs = []
    
    os.makedirs(args.result_path, exist_ok=True)
    dir_name = 'result'
    out_counter = len([None for out in os.listdir(args.result_path) if dir_name in out])
    result_path = os.path.join(args.result_path, dir_name + '_' + str(out_counter + 1))
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        os.makedirs(os.path.join(result_path, 'weights'))
            
    for epoch in range(num_epochs):
        
        # Save configs
        with open(os.path.join(result_path, 'config.json'), 'w') as f:
            f.write(json.dumps(vars(args), indent=4))
        
        t_epoch_start = time.time()
        t_iter_start = time.time()
        
        print("---------")
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print("---------")
        
        for phase in ["train", 'val']:
            if phase == 'train':
                net.train()
                print(' (train) ')
            else:
                if((epoch+1) % 10 == 1):
                    net.eval()
                    print('-------')
                    print(' (val) ')
                else:
                    continue
                    
            for hazy_batch, hazefree_batch in dataloaders_dict[phase]:
                hazy_batch     = hazy_batch.to(device)
                hazefree_batch = hazefree_batch.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    hazefree_outputs = net(hazy_batch)
                    
                    hazefree_loss = criterion(hazefree_outputs, hazefree_batch)
                    
                    loss = hazefree_loss
                    
                    if phase == 'train':
                        loss.backward()
                        
                        optimizer.step()
                        
                        if(iteration % 10 == 0):
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('イテレーション {}||Loss:{:.4f}||10iter:{:.4f} sec.'.format(iteration, loss.item(), duration))
                            t_iter_start = time.time()
                        epoch_train_loss += loss.item()
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item()
        t_epoch_finish = time.time()
        epoch_train_loss /= len(dataloaders_dict["train"])
        epoch_val_loss /= len(dataloaders_dict["val"])
        print('-------------')
        print('epoch{}||Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(epoch+1, epoch_train_loss, epoch_val_loss))
        print('timer: {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()
        
        log_epoch = {'epoch':epoch+1, 'train_loss':epoch_train_loss, 'val_loss':epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv(os.path.join(result_path, "log_output.csv"))
        
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        torch.save(net.state_dict(), os.path.join(result_path, 'weights/AODNet_' + str(epoch+1) + '.pth'))


# + jupyter={"outputs_hidden": true}
num_epochs = 51
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs)
# -

if False:
    net.to(device)
    load_path = 'results_AODNet/result_1/weights/HFNet_5.pth'
    load_weights = torch.load(load_path)
    net.load_state_dict(load_weights)


def save_gt_imgs(each_file_dir, \
                 hazy_gt_img_float, \
                 hazefree_gt_img_float \
                ):
    Image.fromarray((hazy_gt_img_float     * 255).astype(np.uint8)).save(os.path.join(each_file_dir, "hazy_gt.png"))
    Image.fromarray((hazefree_gt_img_float * 255).astype(np.uint8)).save(os.path.join(each_file_dir, "hazefree_gt.png"))


def save_predict_imgs(each_file_dir, \
                      hazefree_predict_img_float \
                     ):
    Image.fromarray((hazefree_predict_img_float * 255).astype(np.uint8)).save(os.path.join(each_file_dir, "hazefree_predict.png"))


def show_figure(mode, hazy, hazefree):
    fig = plt.figure()
    fig.suptitle(mode)
    ax_1 = fig.add_subplot(121)
    ax_1.imshow(hazy)
    ax_2 = fig.add_subplot(122)
    ax_2.imshow(hazefree)
    plt.show()


def calculate_psnr_and_ssim(psnrs, ssims, gt, predict):
    gt = np.where(gt > 1, 1, gt)
    gt = np.where(gt < 0, 0, gt)
    predict = np.where(predict > 1, 1, predict)
    predict = np.where(predict < 0, 0, predict)
    
    psnr = peak_signal_noise_ratio(gt, predict)
    ssim = structural_similarity(gt, predict, multichannel=True)
    print("psnr : ", psnr)
    print("ssim : ", ssim)
    psnrs.append(psnr)
    ssims.append(ssim)
    print(np.mean(psnrs))
    print(np.mean(ssims))
    return psnrs, ssims


def test():
    test_files = sorted(glob.glob('../data/main/val/*.npy'))
    image_results_dir = 'image_results/AODNet'
    psnrs = []
    ssims = []
    for file in tqdm(test_files):
        filename = os.path.splitext(os.path.basename(file))[0]
        each_file_dir = os.path.join(image_results_dir, filename)
        os.makedirs(each_file_dir, exist_ok=True)
        
        data = np.load(file, allow_pickle=True)
        hazy_batch = torch.FloatTensor(data[0]).to(device).unsqueeze(0)
        
        hazy_gt_img_float     = data[0].transpose(1, 2, 0)
        hazefree_gt_img_float = data[1].transpose(1, 2, 0)
        
        save_gt_imgs(each_file_dir, hazy_gt_img_float ** (1.0/2.2), hazefree_gt_img_float ** (1.0/2.2))
        
        show_figure('Ground Truth', hazy_gt_img_float ** (1.0/2.2), hazefree_gt_img_float ** (1.0/2.2))
        
        # ネットワークに入れて予測する
        hazefree_outputs = net(hazy_batch)
        
        hazefree_predict_img_float = hazefree_outputs.detach().to('cpu').numpy().reshape(3, 480, 640).transpose(1, 2, 0)
        
        save_predict_imgs(each_file_dir, hazefree_predict_img_float ** (1.0/2.2))
        
        show_figure('Inference', hazy_gt_img_float ** (1.0/2.2), hazefree_predict_img_float ** (1.0/2.2))
        
        psnrs, ssims = calculate_psnr_and_ssim(psnrs, ssims, hazefree_gt_img_float, hazefree_predict_img_float)


plt.rcParams['figure.figsize'] = 20,10
test()
