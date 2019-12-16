import numpy as np
import pickle, json

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
train_gen = data.DataLoader(trainset, batch_size = conf["batch_size"], shuffle = False)

unet = model.UNet(1, 1, conf["start_filters"]).to(device)
criterion = utils.dice_loss
optimizer = optim.Adam(unet.parameters(), lr = conf["lr"])

for epoch in range(conf["epochs"]):
  for batch, labels in train_gen:
    nrrd.write(
    batch, labels = batch.to(device), labels.to(device)
    optimizer.zero_grad()
    logits = unet(batch)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    print(loss.item())
