# imu_compensation.py
import numpy as np

def compensate_positions_body_to_en(
    pos_noimu_EN_m,
    roll_rad_series,
    pitch_rad_series,
    heading_rad_series,
    altitude_m,
    sign_roll_to_right=+1.0,
    sign_pitch_to_fwd=-1.0,
):
    """
    Per-pose (non-integrating) compensation built in the BODY frame, then rotated to EN.

    BODY axes: +Right (y_body), +Forward (x_body).

    For each pose i (absolute 0,0 attitude reference):
        Right_i  = sign_roll_to_right * h * tan(roll_i)
        Fwd_i    = sign_pitch_to_fwd * h * tan(pitch_i)

    Rotate BODY->[N,E] using heading ψ_i:
        [N;E] = [[cosψ, -sinψ],
                 [sinψ,  cosψ]] @ [Fwd; Right]

    Finally:
        pos_comp_EN[i] = pos_noimu_EN[i] - [E_i, N_i]
    """
    P_EN = np.asarray(pos_noimu_EN_m, dtype=float)
    r    = np.asarray(roll_rad_series, dtype=float)
    p    = np.asarray(pitch_rad_series, dtype=float)
    psi  = np.asarray(heading_rad_series, dtype=float)

    N = len(P_EN)
    if N == 0:
        return P_EN.copy()

    # 1) Body-frame compensation components
    right = sign_roll_to_right * altitude_m * np.tan(r)   # +Right
    fwd   = sign_pitch_to_fwd * altitude_m * np.tan(p)    # +Forward

    # 2) Rotate BODY->[N,E] via heading ψ
    c = np.cos(psi); s = np.sin(psi)
    N_comp = c * fwd + (-s) * right
    E_comp = s * fwd +  c   * right

    # 3) Subtract EN compensation from VO baseline
    C_EN = np.column_stack([E_comp, N_comp])  # [E, N]
    return P_EN - C_EN