import numpy as np


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
    H = np.zeros((3, 3))
    A = np.vstack([points[:, 0], np.ones(len(points))]).T
    H[0, 0], H[0, 2] = np.linalg.lstsq(A, points[:, 2])[0]
    B = np.vstack([points[:, 1], np.ones(len(points))]).T
    H[1, 1], H[1, 2] = np.linalg.lstsq(B, points[:, 3])[0]
    H[2, 2] = 1
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
    tX = np.dot(H, X.T).T
    e = np.sqrt((tX - Y) ** 2)
    e = e.sum(axis=1)
    return e


def ransac(data, max_iter=500, tol=100, d_min=5, verbose=False):
    """
    Fits RANSAC

    params
    ------
        data: ndarray
    """
    bestfit = None
    best_inliners = None
    max_d = d_min
    for it in range(max_iter):
        fit_data, test_data = random_partition(2, data.shape[0])
        fit_data = data[fit_data, :]
        test_data = data[test_data]
        fit_H = fit_homography(fit_data)
        error = error_homography(fit_H, test_data)
        inliners = test_data[error < tol]
        if 1:
            if verbose and len(inliners) > d_min:
                print "it %d, error min %f, error max %f, inliners %d" % (
                                                               it,
                                                               error.min(),
                                                               error.max(),
                                                               len(inliners))

        if len(inliners) > max_d:
            bestfit = fit_H
            max_d = len(inliners)
            best_inliners = inliners.copy()

    return best_inliners, bestfit

if __name__ == "__main__":
    H = np.array([[2, 0, 4], [0, 2, 5], [0, 0, 1]])
    A = np.array([[1, 1, 1], [3, 2, 1], [4, 5, 1], [2, 4, 1], [6, 8, 1]])
    r = np.dot(H, A.T)
    a = np.concatenate((A[:, :2], r.T[:, :2]), axis=1)
    b = fit_homography(a)
