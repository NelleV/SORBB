import numpy as np
from sklearn.externals.joblib import Memory

from load import read_gt, get_image, load_data
from descriptors import get_interest_points, compute_boundary_desc
from histograms import compute_visual_words
from retrieval import show_results
from test_search import search, search2
from match import score_results

mem = Memory(cachedir='.')
voc = np.load('/home/nelle/vision/vocabulary_5000.npy')
postings = np.load('./data/postings.npy')
names = np.load('./data/images.npy')

results = []
mAP_search = 0
mAP_scoring = 0
#gen = read_gt()
gen = load_data()

num_query = 0
for i, query in enumerate(gen):
    if i == 10:
        break
    if i % 10 == 0:
        print "computed %d images" % i
    print "Searching for potential matches"
    try:
        #im, mask = get_image(query.keys()[0], test=True)
        im, mask = query
    except IOError:
        try:
            im, mask = get_image(query.keys()[0], test=False)
        except IOError:
            print "couldn't find image %s" % (query.keys()[0], )
            continue

    interest_points = mem.cache(get_interest_points)(mask)
    desc, coords = mem.cache(compute_boundary_desc)(im, mask, interest_points)
    visual_words = mem.cache(compute_visual_words)(desc, voc)
    if visual_words is not None:
        num_query += 1
        search_results = mem.cache(search2)(visual_words,
                                             postings,
                                             max_im=200)

        # get label
        true_label = names[i, 1]
        found = (names[search_results[:, 0].astype(int), 1] == true_label)
        num_true = (names[:300, 1] == true_label).sum()
        AP = sum([float(found[:k].sum()) / (k + 1) * found[k]
                  for k
                  in range(len(found))])
        print "Search 1 %f - found %d on %d" % (float(AP) / min(num_true, 200),
                                                found.sum(),
                                                num_true)
        mAP_search += AP / min(num_true, 200)
        file_name = "result_1_%d.png" % i

#        search_results = mem.cache(score_results)(coords, desc,
#                                                  search_results, names, voc,
#                                                verbose=True)
        show_results(search_results, names, file_name=file_name)
print mAP_search / num_query
print mAP_scoring / num_query
