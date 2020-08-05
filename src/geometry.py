import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from numpy.random import uniform
from py2gmsh import Mesh, Entity, Field
from typing import Optional, Tuple


class AirfoilGeometrySampler:
    def __init__(
        self,
        n_points: int = 50,
        top_left_corner: Tuple[float, float] = (-1, 0.5),
        bottom_right_corner: Tuple[float, float] = (2, -0.5),
    ) -> None:

        # Ranges of parameters describing the airfoil geometry.
        # Values taken from the original paper.
        # Camber parameters' ranges
        self.c_1min, self.c_1max = 0.01, 0.96
        self.c_2min, self.c_2max = 0.02, 0.97
        self.c_3min, self.c_3max = -0.074, 0.247
        self.c_4min, self.c_4max = -0.102, 0.206

        # Thickness parameters' ranges
        self.X_Tmin, self.X_Tmax = 0.2002, 0.4813
        self.T_min, self.T_max = 0.0246, 0.3227
        self.rho_min, self.rho_max = 0.175, 1.4944
        self.beta_min, self.beta_max = 0.1452, 4.8724

        self.n_points = n_points

        self.top_left_corner = top_left_corner
        self.bottom_right_corner = bottom_right_corner

    def _compute_airfoil_geometry(
        self,
        c1: float,
        c2: float,
        c3: float,
        c4: float,
        X_T: float,
        T: float,
        rho_bar: float,
        beta_bar: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        rho = rho_bar * (T / X_T) ** 2
        beta = beta_bar * np.arctan(T / (1 - X_T))

        # Camber line
        x_camber = (
            3 * c1 * self.n_points * (1 - self.n_points) ** 2
            + 3 * c2 * (1 - self.n_points) * self.n_points ** 2
            + self.n_points ** 3
        )
        y_camber = (
            3 * c3 * self.n_points * (1 - self.n_points) ** 2
            + 3 * c4 * (1 - self.n_points) * self.n_points ** 2
        )

        # Assemble linear system to solve for thickness parameters
        A = np.array(
            [
                [np.sqrt(X_T), X_T, X_T ** 2, X_T ** 3, X_T ** 4],
                [0.5 / np.sqrt(X_T), 1, 2 * X_T, 3 * X_T ** 2, 4 * X_T ** 3],
                [0.25, 0.5, 1, 1.5, 2],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ]
        )
        R = np.array([T, 0, -np.tan(beta / 2), np.sqrt(2 * rho), 0])

        t = np.linalg.solve(A, R)

        thickness = (
            t[0] * np.sqrt(x_camber)
            + t[1] * x_camber
            + t[2] * x_camber ** 2
            + t[3] * x_camber ** 3
            + t[4] * x_camber ** 4
        )

        upper_curve = y_camber + 0.5 * thickness
        lower_curve = y_camber - 0.5 * thickness

        return x_camber, upper_curve, lower_curve

    def sample_airfoil_geometry(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        c1 = uniform(self.c_1min, self.c_1max)
        c2 = uniform(self.c_2min, self.c_2max)
        c3 = uniform(self.c_3min, self.c_3max)
        c4 = uniform(self.c_4min, self.c_4max)

        X_T = uniform(self.X_Tmin, self.X_Tmax)
        T = uniform(self.T_min, self.T_max)
        rho_bar = uniform(self.rho_min, self.rho_max)
        beta_bar = uniform(self.beta_min, self.beta_max)

        x_camber, upper_curve, lower_curve = self._compute_airfoil_geometry(
            c1=c1, c2=c2, c3=c3, c4=c4, X_T=X_T, T=T, rho_bar=rho_bar, beta_bar=beta_bar
        )
        return x_camber, upper_curve, lower_curve

    def create_mesh(
        self,
        x_camber: np.ndarray,
        upper_curve: np.ndarray,
        lower_curve: np.ndarray,
        dir_to_save: str,
        filename: str = "geometry",
    ) -> None:

        my_mesh = Mesh()

        top_left = Entity.Point([*self.top_left_corner, 0])
        top_right = Entity.Point(
            [self.bottom_right_corner[0], self.top_left_corner[1], 0]
        )
        bottom_left = Entity.Point(
            [self.top_left_corner[0], self.bottom_right_corner[1], 0]
        )
        bottom_right = Entity.Point([*self.bottom_right_corner, 0])

        # Add domain's corners to the geometry
        my_mesh.addEntity(top_left)
        my_mesh.addEntity(top_right)
        my_mesh.addEntity(bottom_left)
        my_mesh.addEntity(bottom_right)

        # Define the outer boundary
        top_boundary = Entity.Curve([top_left, top_right])
        right_boundary = Entity.Curve([top_right, bottom_right])
        bottom_boundary = Entity.Curve([bottom_right, bottom_left])
        left_boundary = Entity.Curve([bottom_left, top_left])

        domain_outer_boundary = [
            top_boundary,
            right_boundary,
            bottom_boundary,
            left_boundary,
        ]
        my_mesh.addEntities(domain_outer_boundary)

        outer_curve = Entity.CurveLoop(domain_outer_boundary, mesh=my_mesh)

        # Add airfoil points
        airfoil_points = []
        for x, y in zip(x_camber, upper_curve):
            point = Entity.Point([x, y, 0])
            airfoil_points.append(point)
            my_mesh.addEntity(point)

        for x, y in zip(x_camber[::-1][1:-1], lower_curve[::-1][1:-1]):
            point = Entity.Point([x, y, 0])
            airfoil_points.append(point)
            my_mesh.addEntity(point)

        # Discretize the airfoil profile
        intervals = []
        for i in range(len(airfoil_points) - 1):
            point1 = airfoil_points[i]
            point2 = airfoil_points[i + 1]
            interval = Entity.Curve([point1, point2])
            intervals.append(interval)

        # Close the profile using endpoints
        point1 = airfoil_points[-1]
        point2 = airfoil_points[0]
        interval = Entity.Curve([point1, point2])
        intervals.append(interval)

        my_mesh.addEntities(intervals)
        airfoil_profile = Entity.CurveLoop(intervals, mesh=my_mesh)

        # Define interior of the domain
        Entity.PlaneSurface([airfoil_profile, outer_curve], mesh=my_mesh)

        # Adding Coherence option
        my_mesh.Coherence = True

        # Saving geometry as .geo and .stl file
        path_to_save_geo = os.path.join(dir_to_save, f"{filename}.geo")
        path_to_save_stl = os.path.join(dir_to_save, f"{filename}.stl")

        my_mesh.writeGeo(path_to_save_geo)
        subprocess.run(
            [
                "gmsh",
                path_to_save_geo,
                "-2",
                "-o",
                path_to_save_stl,
                "-format",
                "stl",
                "-save_all",
            ]
        )

    def plot_airfoil(
        self, x_camber, upper_curve, lower_curve, filename: Optional[str] = None
    ):

        plt.figure()
        plt.plot(x_camber, lower_curve)
        plt.plot(x_camber, upper_curve)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlim(0, 1)
        plt.ylim(-0.25, 0.25)
        if filename is not None:
            plt.savefig(filename)
        plt.show()