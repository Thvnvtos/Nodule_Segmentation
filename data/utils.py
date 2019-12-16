import numpy as np
import torch
import nrrd, scipy.ndimage

def dice_loss(logits, labels, eps=1e-7):
  num = 2. * torch.sum(logits * labels)
  denom = torch.sum(logits**2 + labels**2)
  return 1 - torch.mean(num / (denom + eps))

def change_spacing(scan, old_spacing, new_spacing = [1, 1, 1]):
  resize_factor = old_spacing / new_spacing
  return scipy.ndimage.interpolation.zoom(scan, resize_factor, mode = "nearest")
