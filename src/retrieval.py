import numpy as np
from itertools import islice

from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn.metrics.pairwise import euclidean_distances

import load
from histograms import compute_histogram_database, compute_histogram
from ransac_homography import ransac
THRESHOLD = 0.95


def candidates_by_histograms(im, mask, histogram_database, vocabulary):
    query_histogram = compute_histogram(im, mask, vocabulary)
    dists = euclidean_distances(query_histogram, histogram_database)
    return islice(dists[0].argsort(), 0, 200)


def match_descriptors(d1, d2, f1, f2):
    """
    Match descriptors

    params
    ------
        d1: ndarray of descriptors for image1

        d2: ndarray of descriptors for image2

        f1: ndarray of position for image1

        f2: ndarray of position for image2

    returns
    -------
        A n*4 array of coordinates
    """
    # FIXME Make the matching more intelligent. Right now, a point from image1
    # can match from several points of image2. We want the best matches on all
    # side.
    distances = euclidean_distances(d1, d2)
    # Let's do eps1
    N1 = np.array([[x, y] for x, y in enumerate(distances.argmin(axis=1))])
    distances_N1 = distances.min(axis=1)
    for X in N1:
        distances[X[0], X[1]] = distances.max()
        distances_N2 = distances.min(axis=1)

    eps1 = np.zeros(distances_N1.shape,
                   dtype=np.float64)
    eps1 += distances_N1
    eps1 /= distances_N2
    eps1 = eps1 < THRESHOLD

    # Let's do eps0
    N1 = np.array([[x, y] for x, y in enumerate(distances.argmin(axis=0))])
    distances_N1 = distances.min(axis=0)
    for X in N1:
        distances[X[1], X[0]] = distances.max()
        distances_N2 = distances.min(axis=0)

    eps0 = np.zeros(distances_N1.shape,
                   dtype=np.float64)
    eps0 += distances_N1
    eps0 /= distances_N2
    eps0 = eps0 < THRESHOLD

    A0 = distances.argmin(axis=0)
    A1 = distances.argmin(axis=1)
    B0 = np.zeros(distances.shape)
    B0[A0[eps0], np.array(range(A0.shape[0]))[eps0]] = 1
    B0[np.array(range(A1.shape[0]))[eps1], A1[eps1]] += 1
    matches = []
    for i, pos in enumerate(B0):
        for j, element in enumerate(pos):
            if element:
                matches.append((f1[i, 0],
                                f1[i, 1],
                                f2[j, 0],
                                f2[j, 1]))

    return np.array(matches)


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


def score_(desc1, desc2, coords1, coords2, alpha=0.75, beta=0.25):
    """
    Scores
    """
    matched_desc = match_descriptors(np.array(desc1), np.array(desc2),
                                     np.array(coords1), np.array(coords2))
    result = ransac(matched_desc, max_iter=3000, tol=75, d_min=15)
    score = alpha * len(result)
    score += beta * len(result) ** 2 / (len(desc1) * len(desc2))
    return score


def match(results, desc, coords, names):
    """
    """


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

    vocabulary = np.load('./data/vocabulary.npy')

    histogram_database = mem.cache(compute_histogram_database)(vocabulary,
                                                               max_im=5)

    gen = load.load_data()
    im, mask = gen.next()

    candidates = candidates_by_histograms(im,
                                          mask,
                                          histogram_database,
                                          vocabulary)

    print('asd')
