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

def check_torch():
    '''
    Check if Torch is successfully built with GPU support
    '''
    torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
    count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
    device = torch.device('cuda' if cuda_avail else 'cpu')
    print('\n'+'-'*60)
    print('----------------------- VERSION INFO -----------------------')
    print('Torch version: {} | Torch Built with CUDA? {}'.format(torch_version, cuda_avail))
    print('# Device(s) available: {}, Name(s): {}'.format(count, name))
    print('-'*60+'\n')
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
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.convt4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convt3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.convt2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.convt1 = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.upsm = nn.Upsample(scale_factor=2, mode='nearest')

        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        x = self.relu(self.bn3(self.convt4(self.upsm(x))))
        x = self.relu(self.bn2(self.convt3(self.upsm(x))))
        x = self.relu(self.bn1(self.convt2(self.upsm(x))))
        x = self.relu(self.bn0(self.convt1(self.upsm(x))))

        return torch.round(x) #self.soft(x)
    
##################################################
################# DATA HANDLING ##################
##################################################


    
##################################################
################# MODEL TRAINING #################
##################################################

    

##################################################
################ MODEL PREDICTION ################
##################################################



##################################################
################ MODEL INFERENCE ################
##################################################

    

##################################################
################# MAIN FUNCTION ##################
##################################################
if __name__ == '__main__':
    pass

############################################################################
################################### END ####################################
############################################################################