import numpy as np
from itertools import islice

from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn.metrics.pairwise import euclidean_distances

import load
from histograms import compute_histogram_database, compute_histogram


def candidates_by_histograms(im, mask, histogram_database, vocabulary):
    query_histogram = compute_histogram(im, mask, vocabulary)
    dists = euclidean_distances(query_histogram, histogram_database)
    return islice(dists[0].argsort(), 0, 200)


def search(visual_words, postings, max_im=200):
    """
    Search for the best matches in the database

    params
    ------
        visual_words: ndarray
            list of visual words contained in the query.

        postings: ndarray of boolean (n, m)
            inverted index, of n descriptors and m images.

        max_im: int, optional, default: 20
            number of results to return

    returns
    -------
        results: ndarray (., 2)
            first columns containes the index of the images retrieved, the
            second the tfidfs scores
    """
    matches = postings[visual_words].copy()
    tf = np.ones((len(visual_words),)).astype(float) / len(visual_words)
    idf = np.log(postings.shape[1] / matches.sum(axis=1))
    tfidfs = np.dot(tf * idf, matches)
    order = tfidfs.argsort()
    tfidfs.sort()

    # FIXME - should probably use np.concatenate
    results = np.zeros((len(tfidfs), 2))
    results[:, 0] = order
    results[:, 1] = tfidfs
    return results[::-1][:max_im]


def scoring():
    """
    Scores
    """
    pass


def show_results(results, names, title=""):
    """
    Show the results in a nice grid.

    Show only 20 top results

    params
    ------
        results: ndarray (., 2)
            array containing, in the first column, the id of the result, on
            the second, the score of that results

        names: ndarray (.)
            image database.
    """
    fig = plt.figure()

    for i, result in enumerate(results):
        if i > 20:
            break
        image_name = names[result[0]]
        image = load.get_image(image_name)

        ax = fig.add_subplot(5, 4, i + 1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        if len(image.shape) == 3:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=cm.gray)
        ax.set_title("%02d" % result[1])
        ax.imshow(image)
        #ax.set_title("%02d" % result[1])
        ax.set_title(image_name)
    plt.show()


if __name__ == "__main__":
    from sklearn.externals.joblib import Memory
    mem = Memory(cachedir='.')

    vocabulary = np.load('./data/vocabulary.mat')

    histogram_database = mem.cache(compute_histogram_database)(vocabulary,
                                                               max_im=5)

    gen = load.load_data()
    im, mask = gen.next()

    candidates = candidates_by_histograms(im,
                                          mask,
                                          histogram_database,
                                          vocabulary)

    print('asd')
