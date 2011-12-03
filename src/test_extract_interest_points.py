import load
import matplotlib.pyplot as plt
from descriptors import get_interest_points

gen = load.load_data()
_, _ = gen.next()
im, mask = gen.next()
points = get_interest_points(mask, min_dist=40)

plt.figure()
plt.imshow(im)
plt.scatter(points[:, 1], points[:, 0])
