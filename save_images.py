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