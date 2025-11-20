"""Geometric domains for PINNs."""

__all__ = ["Geometry", "Interval", "Circle", "Disk", "Ellipse", "Rectangle", "TimeDomain"]

import abc
import numpy as np
from .sampler import sample


class Geometry(abc.ABC):
    """Abstract base class for geometries."""
    
    def __init__(self, dim):
        self.dim = dim
        self.idstr = type(self).__name__

    @abc.abstractmethod
    def inside(self, x):
        """Check if x is inside the geometry (including the boundary)."""

    @abc.abstractmethod
    def on_boundary(self, x):
        """Check if x is on the geometry boundary."""

    @abc.abstractmethod
    def random_points(self, n, random="pseudo"):
        """Compute the random point locations in the geometry."""

    @abc.abstractmethod
    def random_boundary_points(self, n, random="pseudo"):
        """Compute the random point locations on the boundary."""

    def uniform_points(self, n, boundary=True):
        """Compute the equispaced point locations in the geometry."""
        print(f"Warning: {self.idstr}.uniform_points not implemented. Use random_points instead.")
        return self.random_points(n)

    def uniform_boundary_points(self, n):
        """Compute the equispaced point locations on the boundary."""
        print(f"Warning: {self.idstr}.uniform_boundary_points not implemented. Use random_boundary_points instead.")
        return self.random_boundary_points(n)

    def boundary_normal(self, x):
        """Compute the unit normal at x for Neumann or Robin boundary conditions."""
        raise NotImplementedError(f"{self.idstr}.boundary_normal to be implemented")

    def union(self, other):
        """Union of two geometries."""
        from .ops import Union
        return Union(self, other)

    def __or__(self, other):
        """Union operator |"""
        return self.union(other)

    def difference(self, other):
        """Difference of two geometries."""
        from .ops import Difference
        return Difference(self, other)

    def __sub__(self, other):
        """Difference operator -"""
        return self.difference(other)

    def intersection(self, other):
        """Intersection of two geometries."""
        from .ops import Intersection
        return Intersection(self, other)

    def __and__(self, other):
        """Intersection operator &"""
        return self.intersection(other)

    def __mul__(self, other):
        """Cross product operator *"""
        from .ops import CrossProduct
        return CrossProduct(self, other)


class Interval(Geometry):
    """1D interval [left, right]."""
    
    def __init__(self, left, right):
        super().__init__(dim=1)
        self.left, self.right = left, right
        self.length = right - left

    def inside(self, x):
        return np.logical_and(self.left <= x[:, 0], x[:, 0] <= self.right)

    def on_boundary(self, x):
        return np.logical_or(
            np.abs(x[:, 0] - self.left) < 1e-10,
            np.abs(x[:, 0] - self.right) < 1e-10
        )

    def random_points(self, n, random="pseudo"):
        x = sample(n, 1, random)
        return self.left + self.length * x

    def uniform_points(self, n, boundary=True):
        return np.linspace(self.left, self.right, num=n, endpoint=boundary)[:, None]

    def random_boundary_points(self, n, random="pseudo"):
        if n == 2:
            return np.array([[self.left], [self.right]])
        return np.random.choice([self.left, self.right], n)[:, None]

    def uniform_boundary_points(self, n):
        if n == 1:
            return np.array([[self.left]])
        xl = np.full((n // 2, 1), self.left)
        xr = np.full((n - n // 2, 1), self.right)
        return np.vstack((xl, xr))


class Rectangle(Geometry):
    """2D rectangle domain.
    
    Args:
        xmin, xmax: x-axis bounds
        ymin, ymax: y-axis bounds
    """

    def __init__(self, xmin, xmax, ymin, ymax):
        super().__init__(dim=2)
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.width = xmax - xmin
        self.height = ymax - ymin
        self.perimeter = 2 * (self.width + self.height)
        self.area = self.width * self.height

    def inside(self, x):
        return np.logical_and(
            np.logical_and(self.xmin <= x[:, 0], x[:, 0] <= self.xmax),
            np.logical_and(self.ymin <= x[:, 1], x[:, 1] <= self.ymax)
        )

    def on_boundary(self, x):
        tol = 1e-10
        on_left = np.abs(x[:, 0] - self.xmin) < tol
        on_right = np.abs(x[:, 0] - self.xmax) < tol
        on_bottom = np.abs(x[:, 1] - self.ymin) < tol
        on_top = np.abs(x[:, 1] - self.ymax) < tol
        
        return np.logical_or(
            np.logical_or(on_left, on_right),
            np.logical_or(on_bottom, on_top)
        )

    def random_points(self, n, random="pseudo"):
        z = sample(n, 2, random)
        x = self.xmin + self.width * z[:, 0]
        y = self.ymin + self.height * z[:, 1]
        return np.column_stack((x, y))

    def random_boundary_points(self, n, random="pseudo"):
        p = sample(n, 1, random).flatten() * self.perimeter
        
        x_coords = np.where(
            p < self.width, 
            self.xmin + p,
            np.where(
                p < self.width + self.height,
                self.xmax,
                np.where(
                    p < 2 * self.width + self.height,
                    self.xmax - (p - self.width - self.height),
                    self.xmin
                )
            )
        )
        
        y_coords = np.where(
            p < self.width,
            self.ymin,
            np.where(
                p < self.width + self.height,
                self.ymin + (p - self.width),
                np.where(
                    p < 2 * self.width + self.height,
                    self.ymax,
                    self.ymax - (p - 2 * self.width - self.height)
                )
            )
        )
        
        return np.column_stack((x_coords, y_coords))

    def boundary_normal(self, x):
        """Compute outward normal vectors."""
        tol = 1e-10
        normals = np.zeros_like(x)
        
        # Left boundary: normal = (-1, 0)
        left_mask = np.abs(x[:, 0] - self.xmin) < tol
        normals[left_mask] = [-1, 0]
        
        # Right boundary: normal = (1, 0)
        right_mask = np.abs(x[:, 0] - self.xmax) < tol
        normals[right_mask] = [1, 0]
        
        # Bottom boundary: normal = (0, -1)
        bottom_mask = np.abs(x[:, 1] - self.ymin) < tol
        normals[bottom_mask] = [0, -1]
        
        # Top boundary: normal = (0, 1)
        top_mask = np.abs(x[:, 1] - self.ymax) < tol
        normals[top_mask] = [0, 1]
        
        return normals


class Disk(Geometry):
    """2D disk domain."""
    
    def __init__(self, center, radius):
        super().__init__(dim=2)
        self.center = np.array(center)
        self.radius = radius

    def inside(self, x):
        return np.linalg.norm(x - self.center, axis=-1) <= self.radius

    def on_boundary(self, x):
        return np.abs(np.linalg.norm(x - self.center, axis=-1) - self.radius) < 1e-10

    def random_points(self, n, random="pseudo"):
        rng = sample(n, 2, random)
        r, theta = rng[:, 0], 2 * np.pi * rng[:, 1]
        x = np.sqrt(r) * np.cos(theta)
        y = np.sqrt(r) * np.sin(theta)
        return self.radius * np.column_stack((x, y)) + self.center

    def random_boundary_points(self, n, random="pseudo"):
        u = sample(n, 1, random)
        theta = 2 * np.pi * u
        x = np.cos(theta)
        y = np.sin(theta)
        return self.radius * np.column_stack((x, y)) + self.center

    def uniform_boundary_points(self, n):
        theta = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
        x = np.cos(theta)
        y = np.sin(theta)
        return self.radius * np.column_stack((x, y)) + self.center

    def boundary_normal(self, x):
        """Compute outward normal vectors."""
        diff = x - self.center
        norm = np.linalg.norm(diff, axis=-1, keepdims=True)
        return diff / norm


class Circle(Geometry):
    """1D circle (circumference only)."""
    
    def __init__(self, center, radius):
        super().__init__(dim=1)
        self.center = np.array(center)
        self.radius = radius

    def inside(self, x):
        # Circle is 1D manifold, no interior
        return np.zeros(len(x), dtype=bool)

    def on_boundary(self, x):
        return np.abs(np.linalg.norm(x - self.center, axis=-1) - self.radius) < 1e-10

    def random_points(self, n, random="pseudo"):
        theta = 2 * np.pi * sample(n, 1, random)
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        return np.column_stack((x, y))

    def uniform_points(self, n, boundary=True):
        theta = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        return np.column_stack((x, y))

    def random_boundary_points(self, n, random="pseudo"):
        return self.random_points(n, random)


class Ellipse(Geometry):
    """2D ellipse domain."""

    def __init__(self, center, semimajor, semiminor):
        super().__init__(dim=2)
        self.center = np.array(center)
        self.semimajor = semimajor
        self.semiminor = semiminor

    def inside(self, x):
        u = (x[:, 0] - self.center[0]) / self.semimajor
        v = (x[:, 1] - self.center[1]) / self.semiminor
        return u**2 + v**2 <= 1

    def on_boundary(self, x):
        u = (x[:, 0] - self.center[0]) / self.semimajor
        v = (x[:, 1] - self.center[1]) / self.semiminor
        return np.abs(u**2 + v**2 - 1) < 1e-10

    def random_points(self, n, random="pseudo"):
        z = sample(n, 2, random)
        r, theta = z[:, 0], 2 * np.pi * z[:, 1]
        x = self.center[0] + np.sqrt(r) * self.semimajor * np.cos(theta)
        y = self.center[1] + np.sqrt(r) * self.semiminor * np.sin(theta)
        return np.column_stack((x, y))

    def random_boundary_points(self, n, random="pseudo"):
        theta = 2 * np.pi * sample(n, 1, random).flatten()
        x = self.center[0] + self.semimajor * np.cos(theta)
        y = self.center[1] + self.semiminor * np.sin(theta)
        return np.column_stack((x, y))

    def uniform_boundary_points(self, n):
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = self.center[0] + self.semimajor * np.cos(theta)
        y = self.center[1] + self.semiminor * np.sin(theta)
        return np.column_stack((x, y))

    def boundary_normal(self, x):
        """Compute outward normal vectors."""
        # For ellipse: normal is proportional to gradient of (x/a)^2 + (y/b)^2
        dx = (x[:, 0] - self.center[0]) / (self.semimajor ** 2)
        dy = (x[:, 1] - self.center[1]) / (self.semiminor ** 2)
        normals = np.column_stack((dx, dy))
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals / norm


class TimeDomain(Interval):
    """Time domain for time-dependent problems."""
    
    def __init__(self, t0, t1):
        super().__init__(t0, t1)
        self.t0 = t0
        self.t1 = t1
