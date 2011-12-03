import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from skimage.draw import bresenham
from sklearn.externals.joblib import Memory
from sklearn.metrics.pairwise import euclidean_distances

from load import load_data
from descriptors import get_interest_points, compute_boundary_desc
from histograms import compute_visual_words
from ransac_homography import ransac

THRESHOLD = 0.99


def match_descriptors(d1, d2, f1, f2):
    """
    Match descriptors

    params
    ------
        d1: ndarray of descriptors for image1

        d2: ndarray of descriptors for image2

        f1: ndarray of position for image1

        f2: ndarray of position for image2

    returns
    -------
        A n*4 array of coordinates
    """
    # FIXME Make the matching more intelligent. Right now, a point from image1
    # can match from several points of image2. We want the best matches on all
    # side.
    distances = euclidean_distances(d1, d2)
    mean_norm = float((d1 ** 2).sum() +
                      (d2 ** 2).sum()) / (d2.shape[0] * d2.shape[1] +
                                        d1.shape[0] * d2.shape[1])
    distances /= mean_norm
    # the nearest neighbour is the one for which the euclidean distance is the
    # smallest
#    N1 = np.array([[x, y] for x, y in enumerate(distances.argmin(axis=1))])
#    distances_N1 = distances.min(axis=1)
#    for X in N1:
#        distances[X[0], X[1]] = distances.max()
#    distances_N2 = distances.min(axis=1)
#
#    # FIXME could use float32...
#    eps = np.zeros(distances_N1.shape, dtype=np.float64)
#    eps += distances_N1
#    eps /= distances_N2
#    eps = eps < THRESHOLD
#
    A0 = distances.argmin(axis=0)
    A1 = distances.argmin(axis=1)
    B = np.zeros(distances.shape)
    B[A0, range(A0.shape[0])] += distances.min(axis=0)
    B[range(A1.shape[0]), A1] += distances.min(axis=1)
    matches = []
    matches_d = []
    for i, pos in enumerate(B):
        for j, element in enumerate(pos):
            if element:
                matches.append((f1[i, 0],
                                f1[i, 1],
                                f2[j, 0],
                                f2[j, 1]))
                #matches_d.append(N1[i])

    return matches, matches_d


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


mem = Memory(cachedir='.')

vocabulary = np.load('./data/vocabulary.mat')
gen = load_data()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()

im1, mask1 = gen.next()
_, _ = gen.next()

im2, mask2 = gen.next()
interest_points = mem.cache(get_interest_points)(mask1)
desc1, coords1 = mem.cache(compute_boundary_desc)(im1,
                                              mask1,
                                              interest_points)
voc1 = vocabulary[compute_visual_words(desc1, vocabulary)]

interest_points = mem.cache(get_interest_points)(mask2)
desc2, coords2 = mem.cache(compute_boundary_desc)(im2,
                                               mask2,
                                               interest_points)
voc2 = vocabulary[compute_visual_words(desc2, vocabulary)]

# Use, as for a sift matching a nearest neighbour /  second nearest neighbour
# matching.
A, _ = match_descriptors(np.array(desc1), np.array(desc2),
                         np.array(coords1), np.array(coords2))
el = ransac(np.array(A), max_iter=1000, tol=10)
show_matched_desc(im1, im2, np.array(el[0]))
