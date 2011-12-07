import numpy as np

from sklearn.externals.joblib import Memory

from load import load_data
from descriptors import compute_boundary_desc, get_interest_points
from histograms import compute_visual_words
from retrieval import search, show_results

mem = Memory(cachedir='.')
postings = np.load('./data/postings.npy')
vocabulary = np.load('/home/nelle/vision/vocabulary_5000.npy')
image_names = np.load('./data/images.npy')

gen = load_data()
im, mask = gen.next()

interest_points = mem.cache(get_interest_points)(mask)
descriptors, _ = mem.cache(compute_boundary_desc)(im, mask, interest_points)
vw = compute_visual_words(descriptors, vocabulary)

# results = search(vw, postings)
def search(vw, postings, max_im=200):
    res = postings[vw].astype(bool).sum(axis=0)
    # FIXME - should probably use np.concatenate
    order = res.argsort()
    res.sort()

    results = np.zeros((len(res), 2))
    results[:, 0] = order
    results[:, 1] = res
    return results[::-1][:max_im]

def search2(vw, postings, max_im=200):
    database_tf = postings / postings.sum(axis=0)
    database_tf[np.isnan(database_tf)] = 0
    idf = np.log(postings.shape[1] / postings.sum(axis=1))
    hist, _ = np.histogram(vw, bins=np.arange(len(postings) + 1))
    query_tfidf = hist.astype(float) / len(vw) * idf
    query_tfidf[np.isnan(query_tfidf)] = 0
    database_tfidf = database_tf * idf.reshape(len(idf), 1)
    database_tfidf[np.isnan(database_tfidf)] = 0

    res = np.dot(query_tfidf, database_tfidf)
    res /= (query_tfidf ** 2).sum() * (database_tfidf ** 2).sum(axis=0)
    res[np.isnan(res)] = 0
    # FIXME - should probably use np.concatenate
    order = res.argsort()
    res.sort()

    results = np.zeros((len(res), 2))
    results[:, 0] = order
    results[:, 1] = res
    return results[::-1][:max_im]

results = search2(vw, postings)
#show_results(results, image_names)

# Awesome ! We're finding the the first image as best result !
