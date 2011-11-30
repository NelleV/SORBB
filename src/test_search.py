import numpy as np

from sklearn.externals.joblib import Memory

from load import load_data
from descriptors import compute_boundary_desc, get_interest_points
from histograms import compute_visual_words
from retrieval import search

mem = Memory(cachedir='.')
postings = np.load('./data/postings.mat')
vocabulary = np.load('./data/vocabulary.mat')
gen = load_data()
im, mask = gen.next()

interest_points = mem.cache(get_interest_points)(mask)
descriptors = mem.cache(compute_boundary_desc)(im, mask, interest_points)
vw = compute_visual_words(descriptors, vocabulary)
results = search(vw, postings)

# Awesome ! We're finding the the first image as best result !
