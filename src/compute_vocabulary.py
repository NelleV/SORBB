import numpy as np

from sklearn.externals.joblib import Memory

from vocabulary import compute_vocabulary

NUM_DESC = 10000

mem = Memory(cachedir='.')

descriptors = np.load('./data/descriptors.npy')[:NUM_DESC]

print "compute vocabulary"
vocabulary = compute_vocabulary(descriptors, verbose=True)
vocabulary.dump('./data/vocabulary.mat')
