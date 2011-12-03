import numpy as np

from sklearn.externals.joblib import Memory

from vocabulary import compute_vocabulary

NUM_DESC = None

mem = Memory(cachedir='.')
if NUM_DESC is not None:
    descriptors = np.load('./data/descriptors.npy')[:NUM_DESC]
else:
    descriptors = np.load('./data/descriptors.npy')

# When not computing on the whole database, it's not worth computing to many k
# The ratio is about length of the descriptor * 20 / (number of images)
k = int(float(len(descriptors) / 20))
print "compute vocabulary of size %d" % k
# Compute vocabulary with loose tolerance when testing. The computation is
# then much faster, and still allows correct testing.
vocabulary = compute_vocabulary(descriptors, k=k, verbose=True, tol=10e2)
vocabulary.dump('./data/vocabulary.npy')
