"""Operations on geometries: Union, Intersection, Difference, CrossProduct."""

__all__ = ["Union", "Difference", "Intersection", "CrossProduct"]

import numpy as np
from .geometry import Geometry


class Union(Geometry):
    """Union of two geometries."""
    
    def __init__(self, geom1, geom2):
        if geom1.dim != geom2.dim:
            raise ValueError(
                f"{geom1.idstr} | {geom2.idstr} failed (dimensions do not match)."
            )
        super().__init__(geom1.dim)
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x):
        return np.logical_or(self.geom1.inside(x), self.geom2.inside(x))

    def on_boundary(self, x):
        return np.logical_or(
            np.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x)),
            np.logical_and(self.geom2.on_boundary(x), ~self.geom1.inside(x)),
        )

    def boundary_normal(self, x):
        mask1 = np.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x))
        mask2 = np.logical_and(self.geom2.on_boundary(x), ~self.geom1.inside(x))
        
        normals = np.zeros_like(x)
        if np.any(mask1):
            normals[mask1] = self.geom1.boundary_normal(x[mask1])
        if np.any(mask2):
            normals[mask2] = self.geom2.boundary_normal(x[mask2])
        
        return normals

    def random_points(self, n, random="pseudo"):
        # Sample from bounding box and reject points outside
        x = np.empty(shape=(n, self.dim))
        i = 0
        max_iterations = 100
        iteration = 0
        
        while i < n and iteration < max_iterations:
            # Generate candidate points from both geometries
            n_remaining = n - i
            n1 = n_remaining // 2 + n_remaining % 2
            n2 = n_remaining // 2
            
            tmp1 = self.geom1.random_points(n1, random=random)
            tmp2 = self.geom2.random_points(n2, random=random)
            tmp = np.vstack([tmp1, tmp2])
            
            # Shuffle to mix points from both geometries
            idx = np.random.permutation(len(tmp))
            tmp = tmp[idx]
            
            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
            iteration += 1
        
        return x[:i] if i < n else x

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim))
        i = 0
        max_iterations = 100
        iteration = 0
        
        while i < n and iteration < max_iterations:
            n_remaining = n - i
            geom1_boundary_points = self.geom1.random_boundary_points(n_remaining, random=random)
            geom1_boundary_points = geom1_boundary_points[
                ~self.geom2.inside(geom1_boundary_points)
            ]

            geom2_boundary_points = self.geom2.random_boundary_points(n_remaining, random=random)
            geom2_boundary_points = geom2_boundary_points[
                ~self.geom1.inside(geom2_boundary_points)
            ]

            tmp = np.vstack([geom1_boundary_points, geom2_boundary_points])
            tmp = tmp[np.random.permutation(len(tmp))]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
            iteration += 1
        
        return x[:i] if i < n else x


class Difference(Geometry):
    """Difference of two geometries (geom1 - geom2)."""
    
    def __init__(self, geom1, geom2):
        if geom1.dim != geom2.dim:
            raise ValueError(
                f"{geom1.idstr} - {geom2.idstr} failed (dimensions do not match)."
            )
        super().__init__(geom1.dim)
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x):
        return np.logical_and(self.geom1.inside(x), ~self.geom2.inside(x))

    def on_boundary(self, x):
        return np.logical_or(
            np.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x)),
            np.logical_and(self.geom1.inside(x), self.geom2.on_boundary(x)),
        )

    def boundary_normal(self, x):
        mask1 = np.logical_and(self.geom1.on_boundary(x), ~self.geom2.inside(x))
        mask2 = np.logical_and(self.geom1.inside(x), self.geom2.on_boundary(x))
        
        normals = np.zeros_like(x)
        if np.any(mask1):
            normals[mask1] = self.geom1.boundary_normal(x[mask1])
        if np.any(mask2):
            normals[mask2] = -self.geom2.boundary_normal(x[mask2])
        
        return normals

    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim))
        i = 0
        max_iterations = 100
        iteration = 0
        
        while i < n and iteration < max_iterations:
            tmp = self.geom1.random_points(n - i + 100, random=random)
            tmp = tmp[~self.geom2.inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
            iteration += 1
        
        return x[:i] if i < n else x

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim))
        i = 0
        max_iterations = 100
        iteration = 0
        
        while i < n and iteration < max_iterations:
            n_remaining = n - i
            geom1_boundary_points = self.geom1.random_boundary_points(n_remaining, random=random)
            geom1_boundary_points = geom1_boundary_points[
                ~self.geom2.inside(geom1_boundary_points)
            ]

            geom2_boundary_points = self.geom2.random_boundary_points(n_remaining, random=random)
            geom2_boundary_points = geom2_boundary_points[
                self.geom1.inside(geom2_boundary_points)
            ]

            tmp = np.vstack([geom1_boundary_points, geom2_boundary_points])
            tmp = tmp[np.random.permutation(len(tmp))]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
            iteration += 1
        
        return x[:i] if i < n else x


class Intersection(Geometry):
    """Intersection of two geometries."""
    
    def __init__(self, geom1, geom2):
        if geom1.dim != geom2.dim:
            raise ValueError(
                f"{geom1.idstr} & {geom2.idstr} failed (dimensions do not match)."
            )
        super().__init__(geom1.dim)
        self.geom1 = geom1
        self.geom2 = geom2

    def inside(self, x):
        return np.logical_and(self.geom1.inside(x), self.geom2.inside(x))

    def on_boundary(self, x):
        return np.logical_or(
            np.logical_and(self.geom1.on_boundary(x), self.geom2.inside(x)),
            np.logical_and(self.geom1.inside(x), self.geom2.on_boundary(x)),
        )

    def boundary_normal(self, x):
        mask1 = np.logical_and(self.geom1.on_boundary(x), self.geom2.inside(x))
        mask2 = np.logical_and(self.geom1.inside(x), self.geom2.on_boundary(x))
        
        normals = np.zeros_like(x)
        if np.any(mask1):
            normals[mask1] = self.geom1.boundary_normal(x[mask1])
        if np.any(mask2):
            normals[mask2] = self.geom2.boundary_normal(x[mask2])
        
        return normals

    def random_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim))
        i = 0
        max_iterations = 100
        iteration = 0
        
        while i < n and iteration < max_iterations:
            tmp = self.geom1.random_points(n - i + 100, random=random)
            tmp = tmp[self.geom2.inside(tmp)]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
            iteration += 1
        
        return x[:i] if i < n else x

    def random_boundary_points(self, n, random="pseudo"):
        x = np.empty(shape=(n, self.dim))
        i = 0
        max_iterations = 100
        iteration = 0
        
        while i < n and iteration < max_iterations:
            n_remaining = n - i
            geom1_boundary_points = self.geom1.random_boundary_points(n_remaining, random=random)
            geom1_boundary_points = geom1_boundary_points[
                self.geom2.inside(geom1_boundary_points)
            ]

            geom2_boundary_points = self.geom2.random_boundary_points(n_remaining, random=random)
            geom2_boundary_points = geom2_boundary_points[
                self.geom1.inside(geom2_boundary_points)
            ]

            tmp = np.vstack([geom1_boundary_points, geom2_boundary_points])
            tmp = tmp[np.random.permutation(len(tmp))]

            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
            iteration += 1
        
        return x[:i] if i < n else x


class CrossProduct(Geometry):
    """Cross product (Cartesian product) of two geometries."""
    
    def __init__(self, geom1, geom2):
        super().__init__(dim=geom1.dim + geom2.dim)
        self.geom1 = geom1
        self.geom2 = geom2
        self.split_dim = geom1.dim

    def inside(self, x):
        d = self.split_dim
        return np.logical_and(
            self.geom1.inside(x[:, :d]), 
            self.geom2.inside(x[:, d:])
        )

    def on_boundary(self, x):
        d = self.split_dim
        x1, x2 = x[:, :d], x[:, d:]
        
        # On boundary if on boundary of first OR second (or both)
        return np.logical_or(
            np.logical_and(self.geom1.on_boundary(x1), self.geom2.inside(x2)),
            np.logical_and(self.geom1.inside(x1), self.geom2.on_boundary(x2))
        )

    def boundary_normal(self, x):
        d = self.split_dim
        x1, x2 = x[:, :d], x[:, d:]
        
        normals = np.zeros_like(x)
        
        # Check which boundary the points are on
        mask1 = np.logical_and(self.geom1.on_boundary(x1), self.geom2.inside(x2))
        mask2 = np.logical_and(self.geom1.inside(x1), self.geom2.on_boundary(x2))
        
        if np.any(mask1):
            n1 = self.geom1.boundary_normal(x1[mask1])
            normals[mask1, :d] = n1
        
        if np.any(mask2):
            n2 = self.geom2.boundary_normal(x2[mask2])
            normals[mask2, d:] = n2
        
        return normals

    def random_points(self, n, random="pseudo"):
        x1 = self.geom1.random_points(n, random)
        x2 = self.geom2.random_points(n, random)
        return np.column_stack((x1, x2))

    def random_boundary_points(self, n, random="pseudo"):
        # Half on boundary of geom1, half on boundary of geom2
        n1 = n // 2
        n2 = n - n1
        
        # Boundary of geom1 × interior of geom2
        boundary1 = self.geom1.random_boundary_points(n1, random)
        inside2 = self.geom2.random_points(n1, random)
        x1 = np.column_stack((boundary1, inside2))
        
        # Interior of geom1 × boundary of geom2
        inside1 = self.geom1.random_points(n2, random)
        boundary2 = self.geom2.random_boundary_points(n2, random)
        x2 = np.column_stack((inside1, boundary2))
        
        return np.vstack((x1, x2))
