"""
Kalman filtering, smoothing, and timestamp alignment utilities.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


def kalman_polar(points, snrs, azi_uncertainty, elev_uncertainty,
                 range_uncert_scale=6e-5, angle_uncert_scale=0.01):
    """
    Kalman filter with RTS smoothing in polar coordinates.

    State vector: [r, θ, φ, dr, dθ, dφ].

    Parameters
    ----------
    points : (N, 3) – [range, azimuth, elevation] per frame
    snrs : (N,) – linear SNR per frame (used for range uncertainty)
    azi_uncertainty : (N,) – azimuth spectral spread per frame
    elev_uncertainty : (N,) – elevation spectral spread per frame
    range_uncert_scale : float
    angle_uncert_scale : float

    Returns
    -------
    (N, 6) smoothed states
    """
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.F = np.array([
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ])
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
    ])
    kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=0.01,
                                  block_size=3, order_by_dim=False)
    kf.P *= np.diag([0.01, 0.1, 0.1, 0.01, 0.2, 0.2])
    kf.x = np.append(points[0], [0, 0, 0])

    states, covs = [], []
    for i in range(len(points)):
        kf.R = np.diag([
            (1.0 / snrs[i]) * range_uncert_scale,
            azi_uncertainty[i] * angle_uncert_scale,
            elev_uncertainty[i] * angle_uncert_scale,
        ])
        kf.predict()
        kf.update(points[i])
        states.append(kf.x.copy())
        covs.append(kf.P.copy())

    states = np.array(states)
    covs = np.array(covs)
    smoothed, _, _, _ = kf.rts_smoother(states, covs)
    return smoothed


def abrupt_check(arr):
    """
    Replace outlier angular measurements with median of neighbours.

    A sample is considered an outlier if it jumps more than 5° from its
    predecessor or more than 8° from any of its neighbours.
    """
    arr = arr.copy()
    for col in range(arr.shape[1]):
        for i in range(5, arr.shape[0]):
            neighbours = np.concatenate((arr[i - 13:i, col],
                                         arr[i + 1:i + 6, col]))
            large_jump = np.any(np.abs(neighbours - arr[i, col]) > 8)
            if np.abs(arr[i, col] - arr[i - 1, col]) > 5 or large_jump:
                if i + 15 < arr.shape[0] and i - 20 >= 0:
                    guide = np.concatenate((arr[i - 20:i, col],
                                            arr[i + 1:i + 15, col]))
                    arr[i, col] = np.median(guide)
                else:
                    arr[i, col] = arr[i - 1, col]
    return arr


def gaussian_window(data, window_size):
    """
    Symmetric moving-average smoother.

    Parameters
    ----------
    data : (N, D) array
    window_size : int – full window width
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    num_rows, num_cols = data.shape
    half = window_size // 2
    out = np.empty_like(data)

    for col in range(num_cols):
        for row in range(num_rows):
            start = max(0, row - half)
            end = min(num_rows, row + half + 1)
            w = min(row - start, end - 1 - row)
            out[row, col] = np.mean(data[row - w:row + w + 1, col])
    return out


def timestamp_find(t_rf, t_gt):
    """
    Align two timestamp sequences by finding zero-crossings of their
    difference.  Returns index lists into *t_gt* and *t_rf* respectively.
    """
    t_gt_idx, t_rf_idx = [], []
    for idx_gt, gt_t in enumerate(t_gt):
        delta = t_rf - gt_t
        crossings = [
            i for i, (a, b) in enumerate(zip(delta[:-1], delta[1:]))
            if a * b < 0
        ]
        if crossings:
            t_gt_idx.append(idx_gt)
            t_rf_idx.append(crossings[-1])
    return t_gt_idx, t_rf_idx
