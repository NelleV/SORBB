import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

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
                point = [j, i]
                if not interest_points or \
                   euclidean_distances(point, interest_points).min() > min_dist:
                    interest_points.append([j, i])

    return np.array(interest_points)


if __name__ == "__main__":
    import load
    import matplotlib.pyplot as plt
    gen = load.load_data()
    _, _ = gen.next()
    im, mask = gen.next()
    points = get_interest_points(mask, min_dist=40)

    plt.figure()
    plt.imshow(mask)
    plt.scatter(points[:, 0], points[:, 1])


