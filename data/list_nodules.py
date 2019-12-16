import numpy as np
import matplotlib.pyplot as plt
import nrrd, os, json, pickle
from scipy.ndimage.measurements import center_of_mass
from glob import glob
from tqdm import tqdm

with open("../config.json") as f:
  config = json.load(f)

scans = sorted(os.listdir(config["data_path"]))

list_nods = []

for s in tqdm(scans):
  num_nod = 1
  while(num_nod < 1000):
    nod_path = glob(os.path.join(config["data_path"], s, '*', '*',"Nodule {}*.nrrd".format(num_nod)))
    if len(nod_path) == 0:
      break

    rel_nod_path = os.path.relpath(nod_path[0], config["data_path"])
    nod = np.swapaxes(nrrd.read(nod_path[0])[0], 0, 2)
    c = center_of_mass(nod)
    c = [int(x) for x in c]
    d = int(np.sum(nod)**(1/3))

    if d > 6:
      list_nods.append({"path" : rel_nod_path, "coord" : c, "diameter" : d})

    #plt.imshow(nod[c[0], c[1] - int(2*d) : c[1] + int(2*d), c[2] - int(2*d) : c[2] + int(2*d)])
    #plt.show()
    num_nod += 1

print(len(list_nods))

with open("list_nodules.pickle", "wb") as f:
    pickle.dump(list_nods, f, protocol=pickle.HIGHEST_PROTOCOL)
