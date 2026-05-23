"""
Coordinate conversion and trajectory error metrics.
"""

import numpy as np


def polar_to_cartesian(r, azimuth, elevation):
    """
    Polar → Cartesian.

    Convention: elevation measured from +z axis (0 = zenith, π = nadir).
    """
    x = r * np.sin(elevation) * np.cos(azimuth)
    y = r * np.sin(elevation) * np.sin(azimuth)
    z = r * np.cos(elevation)
    return x, y, z


def cartesian_to_polar(x, y, z):
    """Cartesian → (range, azimuth, elevation)."""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    azimuth = np.arctan(y / x)
    elevation = np.arccos(z / r)
    return r, azimuth, elevation


def get_error(gt_points, est_points):
    """
    Compute per-frame and aggregate trajectory errors.

    Returns
    -------
    mean_error, median_error, total_error : float
    mean_dim_error, median_dim_error : (3,)
    norm_errors : (N,)  – per-frame Euclidean error
    dim_errors  : (N, 3) – per-frame per-axis absolute error
    """
    dim_errors = np.abs(gt_points - est_points)
    norm_errors = np.linalg.norm(dim_errors, axis=1)
    return (
        np.mean(norm_errors),
        np.median(norm_errors),
        np.sum(norm_errors),
        np.mean(dim_errors, axis=0),
        np.median(dim_errors, axis=0),
        norm_errors,
        dim_errors,
    )
