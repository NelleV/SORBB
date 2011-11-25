import numpy as np

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


def get_patch(image, mask, points):
    """
    Creates the patches for each interest points

    params
    ------
       image: ndarray of shape (., ., 3)

       points: ndarray of shape (N, 2)

    returns
    --------

    """
    im = image.mean(axis=2)
    for point in points:
        # Let's first start by creating the patches
        point = points[1, :]
        patch = im[point[0] - 16:point[0] + 16, point[1] - 16:point[1] + 16]
        mask_patch = mask[point[0] - 16:point[0] + 16, point[1] - 16:point[1] + 16]
        yield patch, mask_patch


def occupancy_grid(patch):
    """
    Computes the occupancy grid as descrive in [1]_

    params
    ------
        patch: ndarray of shape 16, 16

    returns
    --------
        feature: list of 4 elements

    references
    ----------
    [1]_ Smooth Object Retrieval using a Bag of Boundaries
    """
    feature = []
    feature.append(patch[:16, :16].sum() / 256)
    feature.append(patch[:16, 16:].sum() / 256)
    feature.append(patch[16:, :16].sum() / 256)
    feature.append(patch[:16, 16:].sum() / 256)

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
        feature = []
        feature.append(hog(patch))
        feature.append(occupancy_grid(patch))
        features.append(np.array(feature).flatten())

    return features




if __name__ == "__main__":
    import load
    import matplotlib.pyplot as plt

    from sklearn.externals.joblib import Memory
    mem = Memory(cachedir='.')

    gen = load.load_data()
    _, _ = gen.next()
    im, mask = gen.next()
    points = mem.cache(get_interest_points)(mask, min_dist=40)

    features = compute_boundary_desc(im, mask, points)




