import numpy as np
from matplotlib import pyplot as plt
from skimage.draw import bresenham


def show_matched_desc(image1, image2, matched_desc):
    """
    Draw lines between matched descriptors of two images

    params
    ------
        image1: ndarray

        image2: ndarray
    """
    # Working on grayscale images
    if (image1.shape) == 3:
        image1 = image1.mean(axis=2)
    if (image2.shape) == 3:
        image2 = image2.mean(axis=2)

    # FIXME doesn't work with images of different size
    image = np.zeros((image1.shape[0],
                      image2.shape[1] + 10 + image2.shape[1]))

    image[:, :image1.shape[1]] = image1
    image[:, 10 + image2.shape[1]:] = image2
    placed_desc = matched_desc.copy()
    placed_desc[:, -1] = matched_desc[:, -1] + 10 + image2.shape[1]
    for el in placed_desc:
        try:
            image[bresenham(el[0], el[1], el[2], el[3])] = 0
        except IndexError:
            pass

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(image)
    plt.show()
