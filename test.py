import numpy as np
from geometry.sampler import sample
from geometry.geometry import Interval, Ellipse, Circle, Rectangle, Disk
from geometry.ops import Union, Intersection, Difference, CrossProduct



ellipse = Ellipse([0,0], 3, 2)
x = np.array([[3, 0], [2, 0], [0, 3], [0, 2], [1, 1], [2, 2], [3, 3]])
print(ellipse.on_boundary(x))