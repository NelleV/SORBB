import numpy as np
from sklearn.externals.joblib import Memory

import load
import retrieval
import retrieval2
from histograms import compute_visual_words
from descriptors import compute_boundary_desc, get_interest_points

ground_truth = load.read_gt()

mem = Memory(cachedir='.', verbose=False)
voc = np.load('./data/vocabulary.npy')
postings = np.load('./data/all_postings.npy')
all_names = np.load('./data/all_names.npy')

def average_precision(search_file_names, positive_file_names, ignore_file_names):
    sum_prec = 0.0
    correct = 0

    for i, search_result in enumerate(search_file_names):
        is_true = (search_result not in ignore_file_names) and (search_result in positive_file_names) 
        if is_true:
            correct += 1
            sum_prec += float(correct) / (i + 1)

    return sum_prec / len(search_file_names)


best_prec = float("-inf")
best_results = None
best_query_file_name = None
best_search_file_names = None

mean_average_precision = 0.0
queries_total = 0
max_im = None
for test_query_index, query in enumerate(ground_truth):
    if test_query_index % 10 == 0:
        print "Computed %d images" % test_query_index
    if max_im and test_query_index == max_im:
        break

    query_file_name = query.keys()[0]
    positive_file_names = set(query.values()[0][0])
    ignore_file_names = set(query.values()[0][1])

    if query_file_name not in all_names:
        continue
    
    im, mask = load.get_image(query_file_name)
    interest_points = mem.cache(get_interest_points)(mask)
    desc, coords = mem.cache(compute_boundary_desc)(im, mask, interest_points)
    visual_words = compute_visual_words(desc, voc)
    if visual_words is None:
        continue

    #search_results = retrieval.search2(visual_words, postings, max_im=20)
    query_document, _ = np.histogram(visual_words, bins=np.arange(len(voc) + 1))
    search_results = retrieval2.search(query_document, max_im=20)
    #search_results2 = mem.cache(search2)(visual_words,postings,max_im=20)
    indices = search_results[:,0].astype(int)
    search_file_names = all_names[indices]
    queries_total += 1
    
    ave_prec = average_precision(search_file_names, positive_file_names, ignore_file_names)
    mean_average_precision += ave_prec
    print "Prec: ", ave_prec
    if ave_prec > best_prec:
        best_query_file_name = query_file_name
        best_search_file_names = search_file_names
        best_prec = ave_prec
        best_results = search_results
    #retrieval.show_results(best_results, best_search_file_names, "%s (%f) mAP: %.2f" % (best_query_file_name, best_prec, mean_average_precision))
    

mean_average_precision /= queries_total
retrieval.show_results(best_results, best_search_file_names, "%s (%f) mAP: %.2f" % (best_query_file_name, best_prec, mean_average_precision))
pass