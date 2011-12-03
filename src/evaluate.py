
import numpy as np

import load
import retrieval
import histograms
from descriptors import compute_boundary_desc, get_interest_points


def perform_evaluation(max_im=None, verbose=False):
    vocabulary = np.load('./data/vocabulary.mat')
    #tf_idf_table = np.load('./data/tf_idf_table.mat')
    #ifreq = np.load('./data/ifreq.mat')
    postings = np.load('./data/postings.mat')

    #running on first 300 queries from training set
    gen = load.load_data()
    test_names = np.load('./data/images.npy')

    best_prec = 0.0
    best_results = 0.0
    k = 0
    mean_average_precision = 0.0
    for test_query_index, (im, mask) in enumerate(gen):
        if verbose:
            if test_query_index % 10 == 0:
                print "Computed %d images" % test_query_index

        if max_im and test_query_index == max_im:
            break

        interest_points = get_interest_points(mask)
        descriptor, _ = compute_boundary_desc(im, mask, interest_points)
        if len(descriptor) == 0:
            continue

        k += 1
        visual_words = histograms.compute_visual_words(descriptor, vocabulary)
        search_results = retrieval.search(visual_words, postings)

        # query_document = histograms.compute_histogram(im, mask, vocabulary)
        # search_results = retrieval2.search(query_document,
        #                                    tf_idf_table,
        #                                    ifreq)

        ave_prec = average_precision(test_query_index,
                                     search_results[:, 0],
                                     test_names)
        mean_average_precision += ave_prec

        if ave_prec > best_prec:
            best_prec = ave_prec
            best_results = search_results

    mean_average_precision /= k
    title = "%s mAP: %.2f" % (test_names[test_query_index],
                              mean_average_precision)
    print "mAP: %.2f" % (mean_average_precision, )
    retrieval.show_results(best_results,
                           test_names,
                           title=title)


def average_precision(test_query_index, search_results, test_names):
    sum_prec = 0.0
    correct = 0

    for i, search_result in enumerate(search_results):
        is_true = label(test_names,
                        search_result) == label(test_names,
                                                test_query_index)
        if is_true:
            correct += 1
            sum_prec += float(correct) / (i + 1)

    return sum_prec / len(search_results)


def label(test_names, index):
    file_name = test_names[index]
    return "_".join(file_name.split("_")[:-1])


if __name__ == "__main__":
    perform_evaluation(max_im=100, verbose=True)
