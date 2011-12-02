import load
import retrieval
import retrieval2
import histograms
from descriptors import compute_boundary_desc, get_interest_points

import numpy as np
from itertools import islice
from sklearn.externals.joblib import Memory

def perform_evaluation():
    mem = Memory(cachedir='.')
    vocabulary = np.load('./data/vocabulary.mat')
    #tf_idf_table = np.load('./data/tf_idf_table.mat')
    #ifreq = np.load('./data/ifreq.mat')
    postings = np.load('./data/postings.mat')
    
    test_queries = list(islice(load.load_data(), 0, 300)) #running on first 300 queries from training set
    test_names = list(load.load_data_names())

    best_prec = 0.0
    best_results = 0.0
    best_query = None
    k = 0
    mean_average_precision = 0.0;
    for test_query_index, (im,mask) in enumerate(test_queries):
        interest_points = mem.cache(get_interest_points)(mask)
        descriptor = mem.cache(compute_boundary_desc)(im, mask, interest_points)
        if len(descriptor) == 0:
            continue
        k+=1
        visual_words = histograms.compute_visual_words(descriptor, vocabulary)
        search_results = retrieval.search(visual_words, postings)

        #query_document = histograms.compute_histogram(im, mask, vocabulary)
        #search_results = retrieval2.search(query_document, tf_idf_table, ifreq)
        ave_prec = average_precision(test_query_index, search_results[:, 0], test_names)
        mean_average_precision += ave_prec
        if ave_prec > best_prec:
            best_prec = ave_prec
            best_results = search_results
            best_query = test_query_index

    mean_average_precision /= k
    retrieval.show_results(best_results, test_names, "%s mAP: %.2f" % (test_names[test_query_index], mean_average_precision))

def average_precision(test_query_index, search_results, test_names):
    sum_prec = 0.0;
    correct = 0
    
    for i in range(len(search_results)):
        is_relevant = label(test_names, int(search_results[i])) == label(test_names, test_query_index)
        if is_relevant:
            correct += 1
            sum_prec += float(correct) / (i+1)

    return sum_prec / len(search_results);

def label(test_names, index):
    file_name = test_names[index]
    return "_".join(file_name.split("_")[:-1])

if __name__ == "__main__":
    perform_evaluation()