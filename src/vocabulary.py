from sklearn.cluster import KMeans

from load import load_data
from descriptors import compute_boundary_desc, get_interest_points


def compute_vocabulary(max_im=None, verbose=False):
    """
    Computes the vocabulary for all the training images

    Returns 250 centroids of descriptors, computed using kmeans on all the
    descriptors

    params
    ------
        max_im: int, default None
            maximum number of images to use to compute the vocabulary

        verbose: boolean, default False

    returns
    -------
        cluster_centers: ndarray
    """
    if verbose:
        print "Computing vocabulary"
    descriptors = compute_descriptors_from_all_training_images(
                            max_im=max_im,
                            verbose=verbose)

    if verbose:
        print "Cleaning output"
    # descriptors = clean(descriptors)

    km = KMeans(k=250)
    km.fit(descriptors)

    return km.cluster_centers_


def compute_descriptors_from_all_training_images(max_im=None, verbose=False):
    def gen(max_im=None, verbose=False):
        for i, (im, mask) in enumerate(load_data()):
            if max_im and i > max_im:
                break

            if verbose:
                print "Compute descriptors for", i
            interest_points = get_interest_points(mask)
            descriptor = compute_boundary_desc(im, mask, interest_points)
            yield descriptor

    return sum(gen(max_im=max_im, verbose=verbose), [])
