import numpy as np

from scipy import linalg


def fit_homography(points):
    """
    Calculate the homography

    params
    -------
        points_1: 2*n matrix, n being 4.
            points on image 1

        points_2: 2*n matric, (here n=4)
            points on image2

    returns
    -------
        H, the homography matrix
    """
    a = []
    for X in points:
        x1 = X[1]
        y1 = X[0]
        x2 = X[3]
        y2 = X[2]

        a_x = np.array([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
        a_y = np.array([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1,
                        y2])
        a.append(a_x)
        a.append(a_y)
    A = np.array(a)
    H = linalg.svd(A)[-1].T[:, -1]
    H.shape = (3, 3)
    H /= H[2, 2]
    return H


def random_partition(n, n_data):
    """
    Create a random partition of the data

    params
    ------
        n,: int
            size of the partition

        n_data: int
            length of the array to partition

    returns
    -------
        idxs1, idxs2: int, int
            indexs of the first partition, and of the second partition
    """
    idxs = np.arange(n_data)
    np.random.shuffle(idxs)
    idxs1 = idxs[:n]
    idxs2 = idxs[n:]
    return idxs1, idxs2


def error_homography(H, data):
    """
    """
    X = np.ones((data.shape[0], 3))
    Y = X.copy()
    X[:, :2] = data[:, :2]
    Y[:, :2] = data[:, 2:]
    tX = np.dot(X, H)
    e = np.sqrt((tX - Y) ** 2)
    e = e.sum(axis=1)
    e /= len(data)
    return e


def ransac(data, max_iter=500, tol=100):
    """
    Fits RANSAC

    params
    ------
        data: ndarray
    """
    bestfit = None
    besterr = 10000000000000
    best_inliners = None
    d = 2
    max_d = 2
    for it in range(max_iter):
        fit_data, test_data = random_partition(4, data.shape[0])
        fit_data = data[fit_data, :]
        test_data = data[test_data]
        fit_H = fit_homography(fit_data)
        error = error_homography(fit_H, test_data)
        inliners = test_data[error < tol]
        if 1:
            if len(inliners) > d:
                print "iteration %d" % it, error.min(), len(inliners), besterr

        err = np.mean(error) / len(inliners)
        if len(inliners) > max_d:
            besterr = err
            bestfit = fit_H
            max_d = len(inliners)
            best_inliners = np.concatenate((fit_data, inliners))
    return best_inliners, bestfit
