import os

import numpy as np

from matplotlib.pyplot import imread

data_train_path = './data/sculptures6k/train'
data_test_path = './data/sculptures6k/test'

masks_train_path = './data/masks/train'
masks_test_path = './data/masks/test'

train_names = './data/sculptures6k_train.txt'
test_names = './data/sculptures6k_test.txt'

def load_data():
    """
    """
    f = open(train_names, 'r')
    for sculpture in f.readlines():
        im = imread(os.path.join(data_train_path, sculpture)[:-1])[::-1]
        calc = imread(os.path.join(
                            masks_train_path,
                            sculpture[:-4] + 'png'))
        yield im, calc

