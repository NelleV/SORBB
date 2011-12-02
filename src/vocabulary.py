from sklearn.cluster import MiniBatchKMeans


def compute_vocabulary(descriptors, k=10000, max_im=None, verbose=False):
    """
    Computes the vocabulary for all the training images

    Returns ``k`` centroids of descriptors, computed using kmeans on all the
    descriptors

    params
    ------
        descriptors: list of ndarrays

        max_im: int, default None
            maximum number of images to use to compute the vocabulary

        verbose: boolean, default False

    returns
    -------
        cluster_centers: ndarray
    """

    km = MiniBatchKMeans(k=k, chunk_size=1500, verbose=verbose)
    km.fit(descriptors)

    return km.cluster_centers_
