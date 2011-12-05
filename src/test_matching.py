import numpy as np
from sklearn.externals.joblib import Memory

from load import load_data
from descriptors import get_interest_points, compute_boundary_desc
from histograms import compute_visual_words
from ransac_homography import ransac
from retrieval import match_descriptors
from draw import show_matched_desc


mem = Memory(cachedir='.')
# FIXME Choose a couple of images, and load them properly
vocabulary = np.load('./data/vocabulary.npy')
gen = load_data()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()
_, _ = gen.next()

im1, mask1 = gen.next()
_, _ = gen.next()

im2, mask2 = gen.next()
interest_points = mem.cache(get_interest_points)(mask1)
desc1, coords1 = mem.cache(compute_boundary_desc)(im1,
                                              mask1,
                                              interest_points)
voc1 = vocabulary[compute_visual_words(desc1, vocabulary)]

interest_points = mem.cache(get_interest_points)(mask2)
desc2, coords2 = mem.cache(compute_boundary_desc)(im2,
                                               mask2,
                                               interest_points)
voc2 = vocabulary[compute_visual_words(desc2, vocabulary)]

# Use, as for a sift matching a nearest neighbour /  second nearest neighbour
# matching.
A = match_descriptors(np.array(desc1), np.array(desc2),
                         np.array(coords1), np.array(coords2))
show_matched_desc(im1, im2, np.array(A))
print "found descriptors %d" % len(A)
el = ransac(np.array(A), max_iter=3000, tol=75, d_min=15)
show_matched_desc(im1, im2, np.array(el[0]))
print "found descriptors %d" % len(el[0])
