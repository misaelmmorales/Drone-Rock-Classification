############################################################################
#    Deep learning-based automated rock classification via high-resolution #
#                    drone-captured core sample imagery                    #
############################################################################
# Author: Domenico M. Crisafulli (github.com/dmc1095)                      #
# Co-author: Misael M. Morales (github.com/misaelmmorales)                 #
# Co-Authors: Dr. Carlos Torres-Verdin                                     #
# Date: 2024                                                               #
############################################################################
# Copyright (c) 2024, Misael M. Morales                                    #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
############################################################################

import os, time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

def check_torch(verbose:bool=True):
    if torch.cuda.is_available():
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
        if verbose:
            print('-'*60)
            print('----------------------- VERSION INFO -----------------------')
            print('Torch version: {} | Torch Built with CUDA? {}'.format(torch_version, cuda_avail))
            print('# Device(s) available: {}, Name(s): {}'.format(count, name))
            print('-'*60)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device
    else:
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        if verbose:
            print('-'*60)
            print('----------------------- VERSION INFO -----------------------')
            print('Torch version: {} | Torch Built with CUDA? {}'.format(torch_version, cuda_avail))
            print('-'*60)
        device = torch.device('cpu')
        return device

##################################################
########### ROCK CLASSIFICATION MODEL ############
##################################################
class RockClassification(nn.Module):
    def __init__(self):
        super(RockClassification, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        self.convt3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.convt2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.convt1 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(kernel_size=2, padding=0)
        self.upsm = nn.Upsample(scale_factor=2, mode='nearest')

        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = self.relu(self.bn2(self.convt3(self.upsm(x))))
        x = self.relu(self.bn1(self.convt2(self.upsm(x))))
        x = self.relu(self.bn0(self.convt1(self.upsm(x))))

        return torch.round(x) #self.soft(x)
    
##################################################
################# DATA HANDLING ##################
##################################################
class CustomDataset(Dataset):
    def __init__(self, input_dir, output_dir, transform=None):
        self.input_dir = input_dir
        self.input_filenames = os.listdir(input_dir)
        self.input_filenames.sort()
        self.output_dir = output_dir
        self.output_filenames = os.listdir(output_dir)
        self.output_filenames.sort()
        self.remap_dict = {0:0, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 10:1}
        self.transform = transform

    def __len__(self):
        return len(self.input_filenames)

    def __getitem__(self, idx):
        input_npy_path  = os.path.join(self.input_dir, self.input_filenames[idx])
        output_npy_path = os.path.join(self.output_dir, self.output_filenames[idx])
        x = np.load(input_npy_path)/255
        input_img = torch.tensor(x).unsqueeze(0).nan_to_num(0).type(torch.float32)
        #output_img = np.vectorize(self.remap_dict.get)(np.load(output_npy_path))
        output_img = np.load(output_npy_path)
        output_img = torch.tensor(output_img).unsqueeze(0).nan_to_num(0).type(torch.int32)
        if self.transform is not None:
            input_img, output_img = self.transform(input_img), self.transform(output_img)
        return input_img, output_img

class PatchTransform:
    def __init__(self, patch_w:int=6, patch_h:int=8):
        self.patch_w = patch_w
        self.patch_h = patch_h

    def __call__(self, img):
        sw = img.shape[-2] // self.patch_w
        sh = img.shape[-1] // self.patch_h
        patches = img.unfold(-2, sw, sw).unfold(-2, sh, sh).reshape(-1, 1, sw, sh)
        return patches
    
class PatchNonzeroFilter:
    def __init__(self, background_class:int=0, verbose:bool=False):
        self.background = background_class
        self.verbose = verbose
    
    def __call__(self, ximg, yimg):
        xmask = torch.sum(ximg, dim=(-3,-2,-1)) != self.background
        xfilt = ximg[xmask]
        ymask = torch.sum(yimg, dim=(-3,-2,-1)) != self.background
        yfilt = yimg[ymask]
        if xfilt.shape[0] != yfilt.shape[0]:
            print('Warning: Input and Output shapes do not match | Filtering with smaller mask...') if self.verbose else None
            mask = ymask if xfilt.shape[0] > yfilt.shape[0] else xmask
        else:
            mask = xmask
        return ximg[mask], yimg[mask], mask
    
class PatchReconstruct:
    def __init__(self, patch_w:int=6, patch_h:int=8):
        self.patch_w = patch_w
        self.patch_h = patch_h

    def __call__(self, img, mask):
        s = img.size(-1)
        b = mask.size(0)
        def repatch(_):
            _ = torch.permute(_, (0,2,1,3,4))
            _ = torch.reshape(_, (b, 1, self.patch_w, self.patch_h, s, s))
            _ = torch.permute(_, (0,1,2,4,3,5))
            _ = torch.reshape(_, (b, 1, s*self.patch_w, s*self.patch_h))
            return _
        xout = torch.zeros((b, self.patch_w*self.patch_h, 1, s, s), dtype=img.dtype)
        xout[mask] = img
        xout = repatch(xout)
        return xout

##################################################
################## MODEL SETUP ###################
##################################################
class DroneRockClass:
    def __init__(self, patch_w:int=6, patch_h:int=8):
        self.device     = check_torch()
        self.input_dir  = 'data/x_images'
        self.output_dir = 'data/y_images'

        self.patch_w, self.patch_h = patch_w, patch_h
        self.patch_transform   = PatchTransform(patch_w=self.patch_w, patch_h=self.patch_h)
        self.patch_filter      = PatchNonzeroFilter()
        self.patch_reconstruct = PatchReconstruct()

        self.model     = RockClassification().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=-3, weight_decay=1e-5)

    def load_data(self, train_percent=0.8, batch_size:int=32):
        self.dataset = CustomDataset(self.input_dir, self.output_dir, transform=self.patch_transform)
        train, test = random_split(self.dataset, [int(train_percent*len(self.dataset)), len(self.dataset)-int(train_percent*len(self.dataset))])
        train, valid = random_split(train, [int(train_percent*len(train)), len(train)-int(train_percent*len(train))])
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
        self.all_loader = DataLoader(self.dataset, batch_size=10, shuffle=False)
        return None

    def train_model(self, epochs:int=301, monitor:int=10):
        train_loss, valid_loss = [], []
        for epoch in range(epochs):
            # training
            epoch_train_loss = []
            self.model.train()
            for i, (x_train, y_train) in enumerate(self.train_loader):
                xf, yf, mask = self.patch_filter(x_train, y_train)
                xf, yf = xf.to(self.device), yf.to(self.device)
                self.optimizer.zero_grad()
                yhat = self.model(xf)
                loss = self.criterion(yhat, yf)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss.append(loss.item())
            train_loss.append(np.mean(epoch_train_loss))
            #validation
            epoch_valid_loss = []
            self.model.eval()
            with torch.no_grad():
                for i, (x_valid, y_valid) in enumerate(self.valid_loader):
                    xvf, yvf, vmask = self.patch_filter(x_valid, y_valid)
                    xvf, yvf = xvf.to(self.device), yvf.to(self.device)
                    yvhat = self.model(xvf)
                    vloss = self.criterion(yvhat, yvf)
                    epoch_valid_loss.append(vloss.item())
            valid_loss.append(np.mean(epoch_valid_loss))
            # monitor
            if epoch % monitor == 0:
                print('Epoch: {} | Train Loss: {:.4f} | Valid Loss: {:.4f}'.format(epoch, train_loss[-1], valid_loss[-1]))
        losses = pd.DataFrame({'train': train_loss, 'valid': valid_loss})
        losses.to_csv('losses.csv')
        torch.save(self.model.state_dict(), 'model.pth')
        return None    
    
    def predict(self):
        k = 0
        for i, (x,y) in tqdm(enumerate(self.all_loader)):
            xfilt, yfilt, mask = self.patch_filter(x, y)
            xfilt, yfilt = xfilt.to(self.device), yfilt.to(self.device)
            ypred = self.model(xfilt)
            yout = self.patch_reconstruct(ypred, mask)
            for j in range(yout.size(0)):
                np.save('data/y_predictions/pimg_{}.npy'.format(k), yout[j].squeeze().cpu().numpy())
                k += 1
        return None

##################################################
################# MAIN FUNCTION ##################
##################################################
if __name__ == '__main__':
    model = DroneRockClass()
    model.load_data()
    model.train_model()
    model.predict()

############################################################################
################################### END ####################################
############################################################################