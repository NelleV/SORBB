from sklearn.externals.joblib import Memory

from load import load_data
from descriptors import compute_boundary_desc, get_interest_points
from vocabulary import compute_vocabulary

NUM_IMAGES = 50

mem = Memory(cachedir='.')

gen = load_data()
descriptors = []
print "Compute descriptors"
for i, (im, mask) in enumerate(gen):
    if i % 10 == 0:
        print "Computed %d images" % i
    if i == NUM_IMAGES:
        break
    interest_points = mem.cache(get_interest_points)(mask)
    descriptor = mem.cache(compute_boundary_desc)(im, mask, interest_points)
    for element in descriptor:
        descriptors.append(element)

print "compute vocabulary"
vocabulary = compute_vocabulary(descriptors)
vocabulary.dump('./data/vocabulary.mat')
