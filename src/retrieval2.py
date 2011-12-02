import numpy as np
from numpy.linalg.linalg import norm
import math

def tf(term, document, freq, tfreq):
    if tfreq[document] == 0:
        return 0
    return float(freq[document, term]) / tfreq[document]

def idf(term, document_count, ifreq):
    return math.log(float(document_count) / (1 + ifreq[term]))

def compute_tf_idf_table():
    documents = np.load("./data/histogram_database.mat")
    document_count = documents.shape[0]
    word_count = documents.shape[1]
        
    freq = np.zeros([document_count, word_count])
    ifreq = np.zeros(word_count)
    tfreq = np.zeros(document_count)
    for term in range(word_count):
        for document in range(len(documents)):
            freq[document, term] = documents[document, term]
    for term in range(word_count):
        ifreq[term] = np.count_nonzero(freq[:,term])
    for document in range(document_count):
        tfreq[document] = np.sum(freq[document,:])

    tf_idf_table = np.zeros([document_count, word_count])
    for document in range(len(documents)):
        for term in range(word_count):
            tf_idf_table[document, term] = tf(term, document, freq, tfreq)*idf(term, document_count, ifreq)

    tf_idf_table.dump("./data/tf_idf_table.mat")
    ifreq.dump("./data/ifreq.mat")

def search(query_document, tf_idf_table, ifreq, max_im = 20):
    def sim(v1, v2):
        if v1.sum() == 0 or v2.sum() == 0:
            return float("-inf")
        return float(np.dot(v1,v2)) / (norm(v1) * norm(v2))
    
    document_count = tf_idf_table.shape[0]
    word_count = tf_idf_table.shape[1]
    freq = query_document
    tfreq = np.array([query_document.sum()])

    query_tfidf = np.zeros(word_count)
    for term in range(word_count):
        tf_ = tf(term, 0, freq, tfreq)
        idf_ = idf(term, document_count, ifreq)
        query_tfidf[term] = tf_*idf_

    tfidfs = np.zeros(document_count)
    for i in range(document_count):
        tfidfs[i] = sim(query_tfidf, tf_idf_table[i])

    order = tfidfs.argsort()
    tfidfs.sort()

    results = np.zeros((len(tfidfs), 2))
    results[:, 0] = order
    results[:, 1] = tfidfs
    return results[::-1][:max_im]

if __name__ == "__main__":
    compute_tf_idf_table()