__all__ = ["Interval", "Circle", "Disk", "Ellipse", "Rectangle"]

import abc
from typing import Union, Literal
import numpy as np
from scipy import spatial
from .sampler import sample

class Geometry(abc.ABC):
    def __init__(self, dim):
        self.dim = dim
        self.idstr = type(self).__name__

    @abc.abstractmethod
    def inside(self, x):
        """Check if x is inside the geometry (including the boundary)."""

    @abc.abstractmethod
    def on_boundary(self, x):
        """Check if x is on the geometry boundary."""

    def distance2boundary(self, x, dirn):
        raise NotImplementedError(
            "{}.distance2boundary to be implemented".format(self.idstr)
        )

    def mindist2boundary(self, x):
        raise NotImplementedError(
            "{}.mindist2boundary to be implemented".format(self.idstr)
        )

    def boundary_constraint_factor(self, x):
        raise NotImplementedError(
            "{}.boundary_constraint_factor to be implemented".format(self.idstr)
        )

    def boundary_normal(self, x):
        """Compute the unit normal at x for Neumann or Robin boundary conditions."""
        raise NotImplementedError(
            "{}.boundary_normal to be implemented".format(self.idstr)
        )

    def uniform_points(self, n, boundary=True):
        """Compute the equispaced point locations in the geometry."""
        print(
            "Warning: {}.uniform_points not implemented. Use random_points instead.".format(
                self.idstr
            )
        )
        return self.random_points(n)

    @abc.abstractmethod
    def random_points(self, n, random="pseudo"):
        """Compute the random point locations in the geometry."""

    def uniform_boundary_points(self, n):
        """Compute the equispaced point locations on the boundary."""
        print(
            "Warning: {}.uniform_boundary_points not implemented. Use random_boundary_points instead.".format(
                self.idstr
            )
        )
        return self.random_boundary_points(n)

    @abc.abstractmethod
    def random_boundary_points(self, n, random="pseudo"):
        """Compute the random point locations on the boundary."""

    def periodic_point(self, x, component):
        """Compute the periodic image of x for periodic boundary condition."""
        raise NotImplementedError(
            "{}.periodic_point to be implemented".format(self.idstr)
        )

    def background_points(self, x, dirn, dist2npt, shift):
        raise NotImplementedError(
            "{}.background_points to be implemented".format(self.idstr)
        )

    def union(self, other):
        """Union."""
        from . import ops
        return ops.Union(self, other)

    def __or__(self, other):
        """Union."""
        from . import ops
        return ops.Union(self, other)

    def difference(self, other):
        """Difference."""
        from . import ops 
        return ops.Difference(self, other)

    def __sub__(self, other):
        """Difference."""
        from . import ops
        return ops.Difference(self, other)

    def intersection(self, other):
        """ops Intersection."""
        from . import ops
        return ops.Intersection(self, other)

    def __and__(self, other):
        """Intersection."""
        from . import ops
        return ops.Intersection(self, other)
    
class Interval(Geometry):
    def __init__(self, left, right):
        super().__init__(dim=1)
        self.left, self.right = left, right

    def inside(self, x):
        return np.logical_and(self.left <= x, x <= self.right).flatten()

    def on_boundary(self, x):
        pass

    def distance2boundary(self, x, dirn):
        pass

    def mindist2boundary(self, x):
        pass

    def boundary_constraint_factor(self, x):
        pass

    def boundary_normal(self, x):
        pass

    def uniform_points(self, n):
        return np.linspace(self.left, self.right, num=n + 1, endpoint=False)[1:, None]

    def random_points(self, n, random="pseudo"):
        x = sample(n, 1, random)
        return (self.right  - self.left) * x + self.left

    def uniform_boundary_points(self, n):
        if n == 1:
            return np.array([[self.left]])
        xl = np.full((n // 2, 1), self.left)
        xr = np.full((n - n // 2, 1), self.right)
        return np.vstack((xl, xr))

    def random_boundary_points(self, n, random="pseudo"):
        if n == 2:
            return np.array([[self.left], [self.right]])
        return np.random.choice([self.left, self.right], n)[:, None]

class Circle(Geometry):
    def __init__(self, center, radius):
        super().__init__(dim=1)
        self.center, self.radius = np.array(center), radius

    def inside(self, x):
        pass

    def on_boundary(self, x):
        return False

    def distance2boundary(self, x, dirn):
        pass

    def mindist2boundary(self, x):
        pass

    def boundary_constraint_factor(self, x):
        pass

    def boundary_normal(self, x):
        pass

    def uniform_points(self, n):
        theta = np.linspace(0, 2*np.pi, n)
        x, y = self.center[0]+self.radius*np.cos(theta), self.center[1]+self.radius*np.sin(theta)
        return np.column_stack((x, y))

    def random_points(self, n, random="pseudo"):
        theta = 2*np.pi*sample(n, 1, random)
        x, y = self.center[0]+self.radius*np.cos(theta), self.center[1]+self.radius*np.sin(theta)
        return np.column_stack((x, y))

    def uniform_boundary_points(self, n):
        pass

    def random_boundary_points(self, n, random="pseudo"):
        pass

class Rectangle(Geometry):
    """
    Args:
        bottomLeft: Coordinate of bottom left corner.
        topRight: Coordinate of top right corner.
    """

    def __init__(self, bottomLeft, topRight):
        super().__init__(dim=2)
        self.bottomLeft = bottomLeft
        self.topRight = topRight
        self.width = self.topRight[0] - self.bottomLeft[0]
        self.height = self.topRight[1] - self.bottomLeft[1]
        self.perimeter = 2 * (self.width + self.height)
        self.area = self.height * self.width

    def inside(self, x):
        u = self.bottomLeft[0] <= x[:,0]
        v = x[:,0] <= self.topRight[0]
        w = self.bottomLeft[1] <= x[:,1]
        z = x[:,1] <= self.topRight[1]
        return np.logical_and(np.logical_and(u, v), np.logical_and(w, z))

    def on_boundary(self, x):
        u = self.bottomLeft[0] == x[:,0]
        v = x[:,0] == self.topRight[0]
        w = self.bottomLeft[1] == x[:,1]
        z = x[:,1] == self.topRight[1]
        return np.logical_and(np.logical_and(u, v), np.logical_and(w, z))

    def random_points(self, n, random="pseudo"):
        z = sample(n, 2, random)
        x, y = self.bottomLeft[0] + self.width * z[:, 0], self.bottomLeft[1] + self.height * z[:, 1]
        return np.column_stack((x, y))

    def uniform_boundary_points(self, n):
        nx, ny = np.ceil(n / self.perimeter * (self.topRight - self.bottomLeft)).astype(int)
        xbot = np.hstack(
            (
                np.linspace(self.bottomLeft[0], self.topRight[0], num=nx, endpoint=False)[
                    :, None
                ],
                np.full([nx, 1], self.bottomLeft[1]),
            )
        )
        yrig = np.hstack(
            (
                np.full([ny, 1], self.topRight[0]),
                np.linspace(self.bottomLeft[1], self.topRight[1], num=ny, endpoint=False)[
                    :, None
                ],
            )
        )
        xtop = np.hstack(
            (
                np.linspace(self.bottomLeft[0], self.topRight[0], num=nx + 1)[1:, None],
                np.full([nx, 1], self.topRight[1]),
            )
        )
        ylef = np.hstack(
            (
                np.full([ny, 1], self.bottomLeft[0]),
                np.linspace(self.bottomLeft[1], self.topRight[1], num=ny + 1)[1:, None],
            )
        )
        x = np.vstack((xbot, yrig, xtop, ylef))
        if n != len(x):
            print(
                "Warning: {} points required, but {} points sampled.".format(n, len(x))
            )
        return x

    def random_boundary_points(self, n, random="pseudo"):
        width = self.width
        height = self.height
        perimeter = self.perimeter
        
        p = sample(n, 1, random) * perimeter
        x_coords = np.where(p < width, self.bottomLeft[0] + p,
                            np.where(p < width + height, self.topRight[0],
                            np.where(p < 2 * width + height, self.topRight[0] - (p - width - height), self.bottomLeft[0])))
        
        y_coords = np.where(p < width, self.topRight[1],
                            np.where(p < width + height, self.topRight[1] - (p - width), self.bottomLeft[1]))
        
        return np.column_stack((x_coords, y_coords))

    def boundary_constraint_factor(self, x):
        pass


class Disk(Geometry):
    """
    DONE
    """
    def __init__(self, center, radius):
        super().__init__(dim=2)
        self.center = np.array(center)
        self.radius = radius

    def inside(self, x):
        return np.linalg.norm(x - self.center, axis=-1) <= self.radius

    def on_boundary(self, x):
        return np.linalg.norm(x - self.center, axis=-1) == self.radius

    def boundary_normal(self, x):
        pass

    def random_points(self, n, random="pseudo"):
        rng = sample(n, 2, random)
        r, theta = rng[:, 0], 2 * np.pi * rng[:, 1]
        x, y = np.cos(theta), np.sin(theta)
        return self.radius * (np.sqrt(r) * np.vstack((x, y))).T + self.center

    def uniform_boundary_points(self, n):
        theta = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
        X = np.vstack((np.cos(theta), np.sin(theta))).T
        return self.radius * X + self.center

    def random_boundary_points(self, n, random="pseudo"):
        u = sample(n, 1, random)
        theta = 2 * np.pi * u
        X = np.hstack((np.cos(theta), np.sin(theta)))
        return self.radius * X + self.center
    
    def boundary_constraint_factor(self, x):
        x1, x2 = (x[:,0:1]-self.center[0])/self.semimajor, (x[:,1:2]-self.center[1])/self.semiminor
        return x1**2 + x2**2 - 1

class Ellipse(Geometry):
    """
    Done
    """

    def __init__(self, center, semimajor, semiminor):
        super().__init__(dim=2)
        self.center = np.array(center)
        self.semimajor = semimajor
        self.semiminor = semiminor

    def on_boundary(self, x):
        u = (x[:,0] - self.center[0])/self.semimajor
        v = (x[:,1] - self.center[1])/self.semiminor
        return u**2 + v**2 == 1

    def inside(self, x):
        u = (x[:,0] - self.center[0])/self.semimajor
        v = (x[:,1] - self.center[1])/self.semiminor
        return u**2 + v**2 <= 1

    def random_points(self, n, random="pseudo"):
        x = sample(n, 2, random)
        r, theta = x[:,0], 2*np.pi*x[:,1]
        x = self.center[0] + np.sqrt(r)*self.semimajor * np.cos(theta)
        y = self.center[1] + np.sqrt(r)*self.semiminor * np.sin(theta)
        return np.column_stack((x, y))

    def uniform_boundary_points(self, n):
        theta = np.linspace(0, 2*np.pi, n)
        x = self.center[0]+self.semimajor * np.cos(theta)
        y = self.center[1]+self.semiminor * np.sin(theta)
        return np.column_stack((x, y))

    def random_boundary_points(self, n, random="pseudo"):
        theta = sample(n, 1, random)
        x = self.center[0]+self.semimajor * np.cos(theta)
        y = self.center[1]+self.semiminor * np.sin(theta)
        np.column_stack((x, y))

    def boundary_constraint_factor(self, x):
        x1, x2 = (x[:,0:1]-self.center[0])/self.semimajor, (x[:,1:2]-self.center[1])/self.semiminor
        return x1**2 + x2**2 - 1
