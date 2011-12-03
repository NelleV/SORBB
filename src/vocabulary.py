from sklearn.cluster import MiniBatchKMeans


def compute_vocabulary(descriptors, k=5000, max_im=None, verbose=False,
                       tol=10e4):
    """
    Computes the vocabulary for all the training images

    Returns ``k`` centroids of descriptors, computed using kmeans on all the
    descriptors

    params
    ------
        descriptors: list of ndarrays

        k: int, optional, default: 5000

        max_im: int, default None
            maximum number of images to use to compute the vocabulary

        verbose: boolean, optional, default: False

        tol: int, optional, default: 10e4
            tolerance used for the minibatch kmeans

    returns
    -------
        cluster_centers: ndarray
    """

    km = MiniBatchKMeans(k=k, chunk_size=2500, verbose=verbose, tol=tol)
    km.fit(descriptors)

    return km.cluster_centers_
