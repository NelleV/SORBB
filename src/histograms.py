import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from sklearn.externals.joblib import Memory

from itertools import islice

from load import load_data
from descriptors import get_interest_points, compute_boundary_desc

mem = Memory(cachedir='.')

def compute_histogram(im, mask, vocabulary):
    words = get_visual_words(im, mask, vocabulary)

    histogram = np.zeros([1, len(vocabulary)])
    for word in words:
        histogram[0, word] += 1

    #histogram = normalize(histogram, 'l1')[0]
    return histogram


def get_visual_words(im, mask, vocabulary):
    interest_points = mem.cache(get_interest_points)(mask)
    descriptor = mem.cache(compute_boundary_desc)(im, mask,interest_points)
    desc_count = len(descriptor)
    words = np.zeros(desc_count)
    if desc_count > 0:
        dists = euclidean_distances(descriptor, vocabulary)
        words = dists.argmin(axis=1)

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


def compute_histogram_database(max_im=1000, word_count = 1000):
    vocabulary = np.load("./data/vocabulary.mat")
    images = list(islice(load_data(), 0, max_im))
    res = np.zeros([len(images), word_count])
    for i, (im, mask) in enumerate(images):
        if i % 10 == 0:
            print "computed %d images" % i
        res[i] = compute_histogram(im, mask, vocabulary)
    
    res.dump("./data/histogram_database.mat")
    return res