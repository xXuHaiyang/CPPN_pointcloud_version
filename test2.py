import numpy as np
N = 1000
per_edge = int(N**(1/3))
s1 = np.linspace(-1, 1, per_edge, endpoint=True)
s2 = np.linspace(-1, 1, per_edge, endpoint=True)
s1, s2 = np.meshgrid(s1, s2)
s3 = np.random.choice([-1, 1], size=s1.shape)
points = np.stack([s1, s2, s3], axis=-1)
print(points.shape)
print(points)
