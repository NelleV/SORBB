"""
Computes the training image database with the labels
"""
import numpy as np
from load import load_data_names

gen = load_data_names()
images = []
labels = []

for i, name in enumerate(gen):
    label = "_".join(name.split("_")[:-1])
    if label not in labels:
        print "Added label %s" % label
        labels.append(label)
    images.append([name, labels.index(label)])


names = np.array(images)
labels = np.array(labels)
names.dump('./data/images.npy')
# Doubt that this will be useful, but who knows
labels.dump('./data/labels.npy')

# Now, let's do the same with test images.
gen = load_data_names(test=True)
images = []
labels = []

for i, name in enumerate(gen):
    label = "_".join(name.split("_")[:-1])
    if label not in labels:
        print "Added label %s" % label
        labels.append(label)
    images.append([name, labels.index(label)])


names = np.array(images)
labels = np.array(labels)
names.dump('./data/test_images.npy')
