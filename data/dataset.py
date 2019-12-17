import numpy as np
import pickle, nrrd, os
from glob import glob
from scipy.ndimage.measurements import center_of_mass

import torch
from torch.utils import data

from . import utils

class Dataset(data.Dataset):

  def __init__(self, data_path, list_nod_path, N):
    self.data_path = data_path
    self.list_nod_path = list_nod_path
    self.N = N
    with open(list_nod_path, "rb") as f:
      self.list_nod = pickle.load(f)

  def __len__(self):
    return self.N

  def __getitem__(self, index):

    nod = self.list_nod[index]
    #print(nod["path"])
    #print(nod["coord"])
    #print(nod["diameter"])
  
    # Load ct file
    path = os.path.join(self.data_path, nod["path"].split('/')[0], '*', '*')
    ct_scan, meta = nrrd.read(glob(os.path.join(path, "*CT.nrrd"))[0])
    ct_scan = np.swapaxes(ct_scan, 0, 2)
    spacing = np.diagonal(meta["space directions"])
    spacing = np.array([abs(spacing[2]), abs(spacing[1]) , abs(spacing[0])])
    nod_mask = np.swapaxes(nrrd.read(os.path.join(self.data_path, nod["path"]))[0], 0, 2)

    # crop nodule box
    z, y, x = nod["coord"]
    d = nod["diameter"]
    ct_scan = ct_scan[z - d:z + d, y - d:y + d, x - d:x + d]
    nod_mask = nod_mask[z - d:z + d, y - d:y + d, x - d:x + d]

    padding = 32
    ct_scan = np.pad(ct_scan, ((padding,padding),(padding,padding),(padding,padding)), "constant", constant_values = ((-3000,-3000),(-3000,-3000),(-3000,-3000)))
    nod_mask = np.pad(nod_mask, ((padding,padding),(padding,padding),(padding,padding)), "constant", constant_values = ((0,0),(0,0),(0,0)))
    
    
    ct_scan = utils.change_spacing(ct_scan, spacing)
    nod_mask = utils.change_spacing(nod_mask, spacing)
    
    z,y,x = [int(x) for x in center_of_mass(nod_mask)]
    
    d = 16
    ct_scan = ct_scan[z - d:z + d, y - d:y + d, x - d:x + d]
    nod_mask = nod_mask[z - d:z + d, y - d:y + d, x - d:x + d]
    return ct_scan[np.newaxis,:], nod_mask[np.newaxis,:]
