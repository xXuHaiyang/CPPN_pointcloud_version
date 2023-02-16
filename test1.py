from tqdm import tqdm
from scipy.spatial import Voronoi, ConvexHull
import numpy as np

def voronoi_volumes(points):
    v = Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # opened regions, i.e., infinite volume (border points)
            vol[i] = 0
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    return vol


x = np.linspace(0, 3, 4)
y = np.linspace(0, 3, 4)
z = np.linspace(0, 3, 4)
coors = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
print(coors)

vols = voronoi_volumes(coors)
print(vols)
coors = np.concatenate((coors, vols.reshape(-1, 1)), axis=1)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

fig = plt.figure()
ax = Axes3D(fig)

for t in coors:
    ax.scatter(t[0], t[1], t[2], c='r', marker='o')
        
plt.show()