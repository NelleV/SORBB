import numpy as np

from sklearn.externals.joblib import Memory
from load import load_data

from descriptors import compute_boundary_desc, get_interest_points
from histograms import compute_visual_words

NUM_IMAGES = None

mem = Memory(cachedir='.')

vocabulary = np.load('./data/vocabulary.npy')
gen = load_data()
res = []

# FIXME needs to lookup the number of images
postings = np.zeros((len(vocabulary), 3170))

for i, (im, mask) in enumerate(gen):
    if i % 10 == 0:
        print "computed %d images" % i
    if NUM_IMAGES is not None and i == NUM_IMAGES:
        break

    interest_points = mem.cache(get_interest_points)(mask)
    descriptor, coords = mem.cache(compute_boundary_desc)(im,
                                                  mask,
                                                  interest_points)
    vw = compute_visual_words(descriptor, vocabulary)
    if vw is not None:
        hist, val = np.histogram(vw, bins=np.arange(len(vocabulary) + 1))
        postings[:, i] = hist

postings.dump('./data/postings.npy')
