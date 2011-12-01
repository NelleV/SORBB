import numpy as np
from load import load_data_names

gen = load_data_names()
names = []

for name in gen:
    names.append(name)

names = np.array(names)
names.dump('./data/images.npy')
