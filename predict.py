import numpy as np
import pickle, json, nrrd

import torch
from torch.utils import data
import torch.optim as optim
import model
from data import *

torch.backends.cudnn.benchmark = True

with open("config.json") as f:
  conf = json.load(f)

device = torch.device("cuda:0")

trainset = dataset.Dataset(data_path = conf["data_path"], list_nod_path = conf["list_nods"], N = conf["N"])

unet = model.UNet(1, 1, conf["start_filters"]).to(device)
unet.load_state_dict(torch.load("./model"))
criterion = utils.dice_loss

x,y = trainset.__getitem__(50)

nrrd.write("ct.nrrd", np.squeeze(x))
nrrd.write("mask.nrrd", np.squeeze(y))

x = torch.tensor(x[np.newaxis,:]).to(device, dtype=torch.float)
y = torch.tensor(y[np.newaxis,:]).to(device, dtype=torch.float)

logits = unet(x)
loss = criterion(logits, y)

logits = np.squeeze(logits.cpu().detach().numpy())
logits[logits < 0.5] = 0
logits[logits > 0] = 1

print(logits.shape)
print(loss.item())
nrrd.write("predicted.nrrd",logits)


