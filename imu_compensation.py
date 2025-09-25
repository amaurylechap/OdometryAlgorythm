# imu_compensation.py
import numpy as np

def compensate_positions_absolute(
    pos_noimu_EN_m,
    roll_rad_series,
    pitch_rad_series,
    altitude_m,
    sign_roll_to_E=+1.0,
    sign_pitch_to_N=+1.0,
):
    """
    Apply per-pose tilt compensation relative to absolute level (0 roll, 0 pitch).

    For each pose i:
        pos_comp[i] = pos_noimu[i] - ( h * tan(roll_i), h * tan(pitch_i) )

    Parameters
    ----------
    pos_noimu_EN_m : (N,2) array
        VO baseline positions in meters [E, N] (uncompensated).
    roll_rad_series, pitch_rad_series : (N,) arrays
        Roll and pitch in radians at each pose time.
    altitude_m : float
        Altitude above ground in meters (constant for now).
    sign_roll_to_E, sign_pitch_to_N : float
        Axis sign mapping (+1/-1) to match conventions.

    Returns
    -------
    pos_comp_EN_m : (N,2) array
        Per-pose compensated positions in meters [E, N].
    """
    P = np.asarray(pos_noimu_EN_m, dtype=float)
    r = np.asarray(roll_rad_series, dtype=float)
    p = np.asarray(pitch_rad_series, dtype=float)

    CE = sign_roll_to_E  * altitude_m * np.tan(r)  # East component
    CN = sign_pitch_to_N * altitude_m * np.tan(p)  # North component

    C = np.column_stack([CE, CN])     # shape (N,2)
    P_comp = P - C
    return P_comp
