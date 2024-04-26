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
import pandas as pd
from scipy.io import loadmat

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

import torch
from torch import nn



############################################################################
################################### END ####################################
############################################################################