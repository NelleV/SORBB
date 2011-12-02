import os

from matplotlib.pyplot import imread

data_train_path = './data/sculptures6k/train'
data_test_path = './data/sculptures6k/test'

masks_train_path = './data/masks/train'
masks_test_path = './data/masks/test'

train_names = './data/sculptures6k_train.txt'
test_names = './data/sculptures6k_test.txt'


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
        im = imread(os.path.join(data_train_path, sculpture)[:-1])[::-1]
        if test:
            try:
                calc = imread(os.path.join(
                                    masks_test_path,
                                    sculpture[:-4] + 'png'))
            except IOError:
                # If the mask is not here, skip this image
                continue
        else:
            try:
                calc = imread(os.path.join(
                                        masks_train_path,
                                        sculpture[:-4] + 'png'))
            except IOError:
                continue
        yield im, calc


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


def get_image(image_name):
    """
    Return the image

    params
    ------
        image_name: string, name of the image

    returns
    -------
        image: ndarray
    """
    image = imread(os.path.join(data_train_path, image_name))[:-1][::-1]
    return image
