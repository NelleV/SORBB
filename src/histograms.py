import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from itertools import *
from load import *
from descriptors import *
from vocabulary import *

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
    descriptors = np.array(clean(descriptors))
    desc_count = len(descriptors)
    words = np.zeros(desc_count)
    if desc_count > 0:
        dists = euclidean_distances(descriptors, vocabulary)
        words = map(lambda i: dists[i].argmin(), range(desc_count))

    return words

def compute_histogram_database(vocabulary):
    data = islice(load_data(), 1, 3)
    res = np.array([compute_histogram(im, mask, vocabulary) for im, mask in data])
    return res