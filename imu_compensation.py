"""
IMU tilt compensation utilities.

Corrects false apparent translations caused by camera tilt
using a simple altitude·tan() projection model.
"""

import numpy as np


def compensate_tilt(dE, dN, roll_rad, pitch_rad, altitude_m):
    """
    Apply tilt compensation to VO per-step displacements.

    Parameters
    ----------
    dE, dN : np.ndarray
        VO-estimated translations per step (East, North) in meters.
    roll_rad, pitch_rad : np.ndarray
        Roll and pitch angles at each step [radians].
        Must be same length as dE, dN.
    altitude_m : float
        Altitude above ground [meters].

    Returns
    -------
    dE_c, dN_c : np.ndarray
        Corrected per-step translations (E, N).
    """
    roll_rad = np.asarray(roll_rad, dtype=float)
    pitch_rad = np.asarray(pitch_rad, dtype=float)

    # Apparent false displacements due to tilt
    false_E = altitude_m * np.tan(roll_rad)
    false_N = altitude_m * np.tan(pitch_rad)

    # Subtract them from VO estimates
    dE_c = dE - false_E
    dN_c = dN - false_N
    return dE_c, dN_c


def integrate_steps(dE_c, dN_c):
    """
    Integrate per-step displacements into cumulative trajectory.
    """
    pos = np.zeros((len(dE_c) + 1, 2), dtype=np.float32)
    pos[1:, 0] = np.cumsum(dE_c)
    pos[1:, 1] = np.cumsum(dN_c)
    return pos
