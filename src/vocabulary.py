from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
from itertools import *
from descriptors import *
from load import *
import math

def compute_vocabulary():
    descriptors = compute_descriptors_from_all_training_images()
    descriptors = clean(descriptors)
        
    km = KMeans(k=250)
    km.fit(descriptors)

    return km.cluster_centers_

def clean(descriptors):
    for i in range(len(descriptors)):
        for j in range(len(descriptors[i])):
            if math.isnan(descriptors[i][j]) or math.isinf(descriptors[i][j]):
                descriptors[i][j] = 0
    return descriptors

def compute_descriptors_from_all_training_images():
    def gen():
        for im,mask in islice(load_data(), 1, 3):
            interestPoints = get_interest_points(mask)
            descriptors = compute_boundary_desc(im, mask, interestPoints)
            yield descriptors
    
    return sum(gen(), [])