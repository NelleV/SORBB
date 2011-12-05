import numpy as np
from sklearn.externals.joblib import Memory

from load import load_data
from descriptors import get_interest_points, compute_boundary_desc
from histograms import compute_visual_words
from retrieval import search, show_results
from match import score_results

NUM_IM = 1
mem = Memory(cachedir='.')
voc = np.load('./data/vocabulary.npy')
postings = np.load('./data/postings.npy')
names = np.load('./data/images.npy')

results = []

gen = load_data()
for i, (im, mask) in enumerate(gen):
    if i % 10 == 0:
        print "computed %d images" % i
    if NUM_IM and i == NUM_IM:
        break
    print "Searching for potential matches"

    interest_points = mem.cache(get_interest_points)(mask)
    desc, coords = mem.cache(compute_boundary_desc)(im, mask, interest_points)
    visual_words = mem.cache(compute_visual_words)(desc, voc)
    search_results = mem.cache(search)(visual_words, postings)

    print "Fine scoring"
    search_results = mem.cache(score_results)(coords, desc,
                                              search_results, names, voc,
                                              verbose=True)
    #show_results(search_results, names)
    results.append(search_results)
