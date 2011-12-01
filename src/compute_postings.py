import numpy as np

from sklearn.externals.joblib import Memory
from load import load_data

from descriptors import compute_boundary_desc, get_interest_points
from histograms import compute_visual_words

NUM_IMAGES = 50

mem = Memory(cachedir='.')

vocabulary = np.load('./data/vocabulary.mat')
gen = load_data()
res = []

# FIXME needs to lookup the number of images
postings = np.zeros((len(vocabulary), 3170))

for i, (im, mask) in enumerate(gen):
    if i % 10 == 0:
        print "computed %d images" % i
    if i == NUM_IMAGES:
        break

    interest_points = mem.cache(get_interest_points)(mask)
    descriptor = mem.cache(compute_boundary_desc)(im,
                                                  mask,
                                                  interest_points)
    vw = compute_visual_words(descriptor, vocabulary)
    if vw is not None:
        postings[vw, i] = True
postings.dump('./data/postings.mat')