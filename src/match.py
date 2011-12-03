from load import get_image
from descriptors import get_interest_points, compute_boundary_desc
from retrieval import score_


def score_results(coords, desc, search_results, names, voc, verbose=False):
    """
    Scores the 200 best results

    params
    ------
        coords:

        desc:

        search_results:

        names: ndarray,
            image database

        voc: ndarray
            vocabulary

        verbose: boolean, optional, default: False
            Make output more verbose

    returns:
        search_results: ndarray
            indxs, scores
    """
    # FIXME doxstring
    for j, (result, score) in enumerate(search_results):
        if verbose:
            print "Scoring %d / %d" % (j, len(search_results))
        im2, mask2 = get_image(names[result, 0])
        interest_points = get_interest_points(mask2)
        desc2, coords2 = compute_boundary_desc(im2,
                                               mask2,
                                               interest_points)
        search_results[j, 1] += score_(desc, desc2, coords, coords2)
    return search_results


def sort(search_results):
    """
    """
