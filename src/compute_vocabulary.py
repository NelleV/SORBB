from sklearn.externals.joblib import Memory

from load import load_data
from descriptors import compute_boundary_desc, get_interest_points
from vocabulary import compute_vocabulary

mem = Memory(cachedir='.')

gen = load_data()
descriptors = []
for im, mask in gen:
    interest_points = mem.cache(get_interest_points)(mask)
    descriptor = mem.cache(compute_boundary_desc)(im, mask, interest_points)
    for element in descriptor:
        descriptors.append(element)

vocabulary = mem.cache(compute_vocabulary)(descriptors)
vocabulary.dump('./data/vocabulary.mat')
