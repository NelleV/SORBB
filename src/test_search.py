import numpy as np

from sklearn.externals.joblib import Memory

from load import load_data
from descriptors import compute_boundary_desc, get_interest_points
from histograms import compute_visual_words
from retrieval import search, show_results

mem = Memory(cachedir='.')
postings = np.load('./data/postings.npy')
vocabulary = np.load('./data/vocabulary.npy')
image_names = np.load('./data/images.npy')

gen = load_data()
im, mask = gen.next()

interest_points = mem.cache(get_interest_points)(mask)
descriptors, _ = mem.cache(compute_boundary_desc)(im, mask, interest_points)
vw = compute_visual_words(descriptors, vocabulary)

results = search(vw, postings)

show_results(results, image_names)

# Awesome ! We're finding the the first image as best result !
