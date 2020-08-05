# Configuration
N_POINTS = 50  # Number of discretization points of lower and upper airfoil profile
N_SAMPLES = 10  # Number of created geometries

import subprocess

import matplotlib.pyplot as plt
import numpy as np

from numpy.random import uniform
from py2gmsh import Mesh, Entity, Field
from typing import Optional, Tuple


class AirfoilGeometrySampler:
    def __init__(self, n_points: int = 50) -> None:

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
