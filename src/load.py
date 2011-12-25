import os
from itertools import chain

from matplotlib.pyplot import imread

data_train_path = './data/sculptures6k/train'
data_test_path = './data/sculptures6k/test'
data_all_path = './data/sculptures6k/all'

masks_train_path = './data/masks/train'
masks_test_path = './data/masks/test'
masks_all_path = './data/masks/all'

train_names = './data/sculptures6k_train.txt'
test_names = './data/sculptures6k_test.txt'

all_names = []
def generate_all_file_names():
    return list(chain(load_data2(test=False), load_data2(test=True)))

def load_data2(test=False):
    """
    Load the data

    params
    ------
        test: boolean, optional
            loads the test data if set to true

    returns
    -------
        (im, calc): tuple of images
    """
    if test:
        f = open(test_names, 'r')
    else:
        f = open(train_names, 'r')
    for sculpture in f.readlines():
        if test:
            try:
                im = imread(os.path.join(data_test_path, sculpture)[:-1])[::-1]
                calc = imread(os.path.join(
                                    masks_test_path,
                                    sculpture[:-4] + 'png'))
            except IOError:
                # If the mask is not here, skip this image
                continue
        else:
            try:
                im = imread(os.path.join(data_train_path, sculpture)[:-1])[::-1]
                calc = imread(os.path.join(
                                        masks_train_path,
                                        sculpture[:-4] + 'png'))
            except IOError:
                continue
        yield sculpture[:-1]

def load_data(test=False):
    """
    Load the data

    params
    ------
        test: boolean, optional
            loads the test data if set to true

    returns
    -------
        (im, calc): tuple of images
    """
    if test:
        f = open(test_names, 'r')
    else:
        f = open(train_names, 'r')
    for sculpture in f.readlines():
        if test:
            try:
                im = imread(os.path.join(data_test_path, sculpture)[:-1])[::-1]
                calc = imread(os.path.join(
                                    masks_test_path,
                                    sculpture[:-4] + 'png'))
            except IOError:
                # If the mask is not here, skip this image
                continue
        else:
            try:
                im = imread(os.path.join(data_train_path, sculpture)[:-1])[::-1]
                calc = imread(os.path.join(
                                        masks_train_path,
                                        sculpture[:-4] + 'png'))
            except IOError:
                continue
        yield im, calc, sculpture[:-1]


def load_data_names(test=False):
    """
    Load the names of the data

    params
    ------
        test: boolean, default: False, optional
            when set to true, read the test dataset

    returns
    --------
        name: string
    """

    if test:
        f = open(test_names, 'r')
    else:
        f = open(train_names, 'r')
    for name in f.readlines():
        yield name[:-1]


def get_image(image_name, test=False):
    """
    Return the image

    params
    ------
        image_name: string, name of the image

    returns
    -------
        image, calc: (ndarray, ndarray)
            returns a tuple of images, one being a sculpture, the other a mask
            Returns None, None if the mask doesn't exist
    """
    try:
        image = imread(os.path.join(data_all_path, image_name))[:-1][::-1]        
        calc = imread(os.path.join(
                            masks_all_path,
                            image_name[:-3] + 'png'))

    except IOError:
        print "IOError %s" % os.path.join(masks_train_path,
                                          image_name[:-3] + 'png')
        return None, None

    return image, calc


def read_gt():
    """
    Read ground truth file

    returns:
        array of dict.
        The dict contains as key the query image, and an array of [truth,
        ignore] as value
    """
    # FIXME - not the best interface
    filename = open('./data/sculptures6k_evaluation/sculptures6k_gt.txt', 'r')
    queries = []
    for i, line in enumerate(filename.readlines()):
        if i == 0:
            continue
        if i % 4 == 1:
            query = line[:-1]
        elif i % 4 == 2:
            continue
        elif i % 4 == 3:
            truth = line.split(' ')
        elif i % 4 == 0:
            ignore = line.split(' ')
            queries.append({query: [truth, ignore]})

    return queries
