import numpy as np
from scipy.misc import imresize

from sklearn.metrics.pairwise import euclidean_distances
from skimage.feature.hog import hog


def get_interest_points(calc, min_dist=40):
    """
    Returns the coordinates of interest points computed on a mask

    params
    ------
        ndarray of booleans

        min_dist: int, optional
            minimum distance between the points of interest.

    returns
    -------
        ndarray:
            array of shape (n, 2) containing the indexes of the interest
            points.

    References
    -----------
    [1] http://www.cs.berkeley.edu/~malik/papers/BMP-shape.pdf (appendix B)
    """
    # Calculates the gradient
    delta = np.diff(calc, axis=-2)[:,:-1] + np.diff(calc, axis=-1)[:-1,:]

    # FIXME there must be a faster way to do that, but lets do it quick and
    # ugly for the sake of the project.
    interest_points = []
    for i in range(0, delta.shape[0]):
        for j in range(0, delta.shape[1]):
            if delta[i, j]:
                point = [i, j]
                if not interest_points or \
                   euclidean_distances(point, interest_points).min() > min_dist:
                    interest_points.append([i, j])

    return np.array(interest_points)


def compute_foreground_area(mask):
    """
    Computes the size of the foreground area

    params
    ------
        mask: ndarray of shape (., .)

    returns
    -------
        min of height and width
    """
    w = mask.argmax(axis=0).max() - mask.argmin(axis=0).min() 
    h = mask.argmax(axis=1).max() - mask.argmin(axis=1).min()
    return min([h, w])


def get_patch(image, mask, points, scales=[1, 4, 16]):
    """
    Creates the patches for each interest points at all desired scales.

    params
    ------
       image: ndarray of shape (., ., 3)

       points: ndarray of shape (N, 2)

    returns
    --------

    """
    im = image.mean(axis=2)
    size = compute_foreground_area(mask) / 20
    # FIXME what happens when the mask goes outside of the image ?
    for scale in scales:
        scale = scale * size
        for point in points:
            patch = im[point[0] - scale:point[0] + scale,
                       point[1] - scale:point[1] + scale]
            mask_patch = mask[point[0] - scale:point[0] + scale,
                              point[1] - scale:point[1] + scale]
            if patch.any():
                yield patch, mask_patch


def occupancy_grid(patch):
    """
    Computes the occupancy grid as descrive in [1]_

    params
    ------
        patch: ndarray of shape N, N

    returns
    --------
        feature: list of 4 elements

    references
    ----------
    [1]_ Smooth Object Retrieval using a Bag of Boundaries
    """
    feature = []
    w, h = patch.shape[0] / 2, patch.shape[1] / 2
    feature.append(patch[:w, :h].sum() / (h * w))
    feature.append(patch[:w, h:].sum() / (h * w))
    feature.append(patch[w:, :h].sum() / (h * w))
    feature.append(patch[:w, h:].sum() / (h * w))

    return feature


def compute_boundary_desc(image, mask, points):
    """
    Compute the boundary descriptors as described in [1]_

    params
    ------
       image: ndarray of shape (., ., 3)

       mask: ndarray of shape (., .)

       points: ndarray of shape (N, 2)

    returns
    -------
       features: list of arrays
    """

    gen = get_patch(image, mask, points)

    features = []
    for patch, mask_patch in gen:
        resized_patch = imresize(patch, (32, 32))
        feature = np.concatenate((hog(resized_patch),
                                  np.array(occupancy_grid(patch))),
                                 axis=0)

        features.append(feature)

    return features




if __name__ == "__main__":
    import load
    import matplotlib.pyplot as plt

    from sklearn.externals.joblib import Memory
    mem = Memory(cachedir='.')

    gen = load.load_data()
    _, _ = gen.next()
    im, mask = gen.next()
    points = mem.cache(get_interest_points)(mask, min_dist=35)

    features = compute_boundary_desc(im, mask, points)




