import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

from load import load_data
from descriptors import get_interest_points, compute_boundary_desc


def compute_histogram(im, mask, vocabulary):
    words = get_visual_words(im, mask, vocabulary)

    histogram = np.zeros([1, len(vocabulary)])
    for word in words:
        histogram[0, word] += 1

    histogram = normalize(histogram, 'l1')[0]
    return histogram


def get_visual_words(im, mask, vocabulary):
    interest_points = get_interest_points(mask)
    descriptors = compute_boundary_desc(im, mask, interest_points)
    desc_count = len(descriptors)
    words = np.zeros(desc_count)
    if desc_count > 0:
        dists = euclidean_distances(descriptors, vocabulary)
        words = map(lambda i: dists[i].argmin(), range(desc_count))

    return words


def compute_visual_words(descriptors, vocabulary):
    """
    Computes the visuals words

    params
    ------
        descriptor: ndarray

        vocabulary: ndarray

    returns
    -------
        ndarray: visual words

    """
    desc_count = len(descriptors)
    if desc_count > 0:
        dists = euclidean_distances(descriptors, vocabulary)
        words = dists.argmin(axis=1)
        return words
    else:
        return None


def compute_histogram_database(vocabulary, max_im=None):
    res = []
    gen = load_data()
    for i, (im, mask) in enumerate(gen):
        if max_im and max_im < i:
            break
        res.append(compute_histogram(im,
                                     mask,
                                     vocabulary))
    return np.array(res)
