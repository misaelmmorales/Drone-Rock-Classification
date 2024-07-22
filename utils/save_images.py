############################################################################
#    Deep learning-based rock classification from core samples obtained    #
#                    from high-resolution drone imagery                    #
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

import os
import numpy as np
from scipy.io import loadmat
import torch
from tqdm import tqdm

'''
Save masked images and masks to disk.
read each img_.mat file in directory and save the image and mask as numpy arrays.
x_images: 'In' masked image 
y_images: 'A_' mask
execution time ~ 30 minutes (10-core Intel i9-10900K)
'''

subimgs, submasks = [], []
for root, dirs, files in os.walk('data'):
    img, mask = [], []
    for f in files:
        if f.startswith('img') and f.endswith('.mat'):
            d = loadmat(os.path.join(root, f), simplify_cells=True)
            imkeys = list(d.keys())
            for j in range(len(imkeys)):
                if imkeys[j].startswith('A_'):
                    q = d[imkeys[j]]
                    mask.append(q)
                    img.append(d['In'] * q)
                imgs, masks = np.array(img), np.array(mask)
            subimgs.append(imgs)
            submasks.append(masks)

k = 0
for i in range(len(subimgs)):
    for j in range(subimgs[i].shape[0]):
        np.save('data/x_images/img_{}.npy'.format(k), subimgs[i][j])
        np.save('data/y_images/img_{}.npy'.format(k), submasks[i][j])
        k += 1

############################################################################
################################### END ####################################
############################################################################

class PatchTransform:
    def __init__(self, patch_w:int=6, patch_h:int=8):
        self.patch_w = patch_w
        self.patch_h = patch_h

    def __call__(self, img):
        img = torch.tensor(img, dtype=torch.float32).view(1, 1, img.shape[-2], img.shape[-1])
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
        return ximg[mask].squeeze(), yimg[mask].squeeze()
    
patch_transform = PatchTransform(patch_w=6, patch_h=8)
nonzero_filter = PatchNonzeroFilter()
nimgs = len(os.listdir('data/x_images'))

all_x_imgs = []
all_y_imgs = []
for i in tqdm(range(nimgs), desc='Processing Images', unit='image'):
    ximg = np.load('data/x_images/img_{}.npy'.format(i))
    yimg = np.load('data/y_images/img_{}.npy'.format(i))
    x_patches = patch_transform(ximg)
    y_patches = patch_transform(yimg)
    ximg, yimg = nonzero_filter(x_patches, y_patches)
    if len(ximg.shape) < 3:
        ximg = ximg.unsqueeze(0)
        yimg = yimg.unsqueeze(0)   
    ximg = ximg/255
    yimg[yimg==255] = 9
    ximg = torch.constant_pad_nd(ximg, (4,4,4,4), value=0)
    yimg = torch.constant_pad_nd(yimg, (4,4,4,4), value=0)
    all_x_imgs.append(ximg)
    all_y_imgs.append(yimg)

x_images = torch.cat(all_x_imgs, dim=0)
y_images = torch.cat(all_y_imgs, dim=0)
print(x_images.shape, y_images.shape)

np.save('data/x_images.npy', np.expand_dims(x_images.detach().numpy(), -1))
np.save('data/y_images.npy', np.expand_dims(y_images.detach().numpy(), -1))