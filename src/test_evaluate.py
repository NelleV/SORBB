import numpy as np
from sklearn.externals.joblib import Memory

from load import load_data
from descriptors import get_interest_points, compute_boundary_desc
from histograms import compute_visual_words
from retrieval import search, show_results, search2
from match import score_results

NUM_IM = 100
mem = Memory(cachedir='.')
voc = np.load('./data/vocabulary.npy')
postings = np.load('./data/postings.npy')
names = np.load('./data/images.npy')

results = []
mAP_search = 0
mAP_scoring = 0
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
    if visual_words is not None:
        search_results = mem.cache(search)(visual_words, postings, max_im=20)
        search_results2 = mem.cache(search2)(visual_words,
                                             postings,
                                             max_im=20)

        # get label
        true_label = names[i, 1]
        found = (names[search_results[:, 0].astype(int), 1] == true_label)
        num_true = (names[:300, 1] == true_label).sum()
        AP = sum([float(found[:k].sum()) / (k + 1) * found[k]
                  for k
                  in range(len(found))])
        print "Search 1 %f - found %d on %d" % (float(AP) / min(num_true, 20),
                                                found.sum(),
                                                num_true)
        mAP_search += AP / min(num_true, 20)

#        search_results = mem.cache(score_results)(coords, desc,
#                                                  search_results, names, voc,
#                                                verbose=True)
        #show_results(search_results, names)

        # get label
        true_label = names[i, 1]
        found = (names[search_results2[:, 0].astype(int), 1] == true_label)
        num_true = (names[:300, 1] == true_label).sum()
        AP = sum([float(found[:k].sum()) / (k + 1) * found[k]
                  for k
                  in range(len(found))])
        print "Search 2 %f, found %d on %d" % (float(AP) / min(num_true, 20),
                                               found.sum(),
                                               num_true)

        true_label = names[i, 1]
        #found = (names[score_results[:, 0].astype(int), 1] == true_label)
        num_true = (names[:300, 1] == true_label).sum()
        AP = sum([float(found[:k].sum()) / (k + 1) * found[k]
                  for k
                  in range(len(found))])
        print "Scoring %f, found %d on %d" % (float(AP) / min(num_true, 20),
                                               found.sum(),
                                               num_true)

        mAP_scoring += AP / min(num_true, 20)
