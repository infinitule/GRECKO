"""Per-track Kalman filter, constant-velocity model.

State: [x, y, vx, vy]. Structured so an IMM bank can wrap multiple
KalmanFilterCV instances later — the filter exposes predict/update as pure
matrix operations on its own state, with no global coupling.

Supports two measurement kinds (matching /proto/sensor_report.schema.json):
- cartesian   : linear update, H = position-selector.
- bearing_only: EKF update linearised about the predicted state relative to
                the observing sensor's position.
"""
from __future__ import annotations

import math

import numpy as np


class KalmanFilterCV:
    NX = 4  # state dimension

    def __init__(self, x0: np.ndarray, P0: np.ndarray, q: float):
        """q: continuous white-acceleration process-noise intensity (m^2/s^3)."""
        self.x = x0.astype(float).copy()
        self.P = P0.astype(float).copy()
        self.q = q

    # -- prediction ---------------------------------------------------------

    def predict(self, dt: float) -> None:
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        # discretised white-acceleration Q
        q = self.q
        dt2, dt3 = dt * dt, dt * dt * dt
        Q = q * np.array([
            [dt3 / 3, 0,       dt2 / 2, 0],
            [0,       dt3 / 3, 0,       dt2 / 2],
            [dt2 / 2, 0,       dt,      0],
            [0,       dt2 / 2, 0,       dt],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    # -- cartesian update ----------------------------------------------------

    @staticmethod
    def _H_cart() -> np.ndarray:
        H = np.zeros((2, 4))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        return H

    def innovation_cart(self, z: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (innovation, innovation covariance S) without updating."""
        H = self._H_cart()
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        return y, S

    def update_cart(self, z: np.ndarray, R: np.ndarray) -> None:
        H = self._H_cart()
        y, S = self.innovation_cart(z, R)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(4) - K @ H
        # Joseph form for numerical stability
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    # -- bearing-only update (EKF) -------------------------------------------

    def innovation_bearing(
        self, bearing: float, R: np.ndarray, sensor_pos: np.ndarray
    ) -> tuple[float, float, np.ndarray]:
        """Return (innovation_scalar, S_scalar, H_row) without updating."""
        dx = self.x[0] - sensor_pos[0]
        dy = self.x[1] - sensor_pos[1]
        r2 = dx * dx + dy * dy
        pred_bearing = math.atan2(dy, dx)
        y = (bearing - pred_bearing + math.pi) % (2 * math.pi) - math.pi
        H = np.zeros((1, 4))
        if r2 > 1e-9:
            H[0, 0] = -dy / r2
            H[0, 1] = dx / r2
        S = float((H @ self.P @ H.T)[0, 0] + R[0, 0])
        return y, S, H

    def update_bearing(self, bearing: float, R: np.ndarray, sensor_pos: np.ndarray) -> None:
        y, S, H = self.innovation_bearing(bearing, R, sensor_pos)
        K = (self.P @ H.T) / S
        self.x = self.x + (K * y).flatten()
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    # -- gating helpers -------------------------------------------------------

    def mahalanobis2_cart(self, z: np.ndarray, R: np.ndarray) -> float:
        y, S = self.innovation_cart(z, R)
        return float(y @ np.linalg.solve(S, y))

    def mahalanobis2_bearing(self, bearing: float, R: np.ndarray, sensor_pos: np.ndarray) -> float:
        y, S, _ = self.innovation_bearing(bearing, R, sensor_pos)
        return float(y * y / S)
