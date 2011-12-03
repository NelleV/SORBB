import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm

from skimage.draw import bresenham


def draw_point(image, x, y):
    """
    Draw points
    """
    image[x][y] = 0
    # image[x][y][1] = 0
    for i in range(3):
        for j in range(3):
            try:
                image[x - i][y - j] = 0
                image[x - i][y + j] = 0
                image[x + i][y - j] = 0
                image[x + i][y + j] = 0
            except IndexError:
                pass


def show_matched_desc(image1, image2, matched_desc):
    """
    Draw lines between matched descriptors of two images

    params
    ------
        image1: ndarray

        image2: ndarray
    """
    # Working on grayscale images
    if len(image1.shape) == 3:
        image1 = image1.mean(axis=2)
    if len(image2.shape) == 3:
        image2 = image2.mean(axis=2)

    # FIXME doesn't work with images of different size
    h = max(image1.shape[0], image2.shape[0])
    image = np.zeros((h,
                      image1.shape[1] + 10 + image2.shape[1]))

    image[:image1.shape[0], :image1.shape[1]] = image1
    image[:image2.shape[0], 10 + image1.shape[1]:] = image2
    placed_desc = matched_desc
    placed_desc[:, -1] = matched_desc[:, -1] + 10 + image1.shape[1]
    for el in placed_desc:
        try:
            draw_point(image, el[0], el[1])
            draw_point(image, el[2], el[3])
            image[bresenham(el[0], el[1], el[2], el[3])] = 0
        except IndexError:
            pass

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(image, cmap=cm.gray)
    plt.show()
