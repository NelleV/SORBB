import numpy as np

from sklearn.externals.joblib import Memory

from vocabulary import compute_vocabulary

NUM_DESC = None

mem = Memory(cachedir='.')
if NUM_DESC is not None:
    descriptors = np.load('./data/descriptors.npy')[:NUM_DESC]
else:
    descriptors = np.load('./data/descriptors.npy')

print "compute vocabulary"
vocabulary = compute_vocabulary(descriptors, verbose=True)
vocabulary.dump('./data/vocabulary.mat')
