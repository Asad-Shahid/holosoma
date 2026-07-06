"""Compute benchmark metrics from a single policy_eval recording.

Reads <policy_dir>/manifest.json plus each scenario's 50 Hz and 200 Hz
NPZ pair; prints a markdown table and optionally writes CSV/JSON.

NPZ arrays carry shape (T, num_envs, ...). Per-env metric values are
computed independently, then aggregated to (mean, std) across envs. The
raw per-env metric values are also dumped to per_env_metrics.json for
paired-comparison analysis. The single-env case (num_envs=1) falls out
naturally: same loop, std=0.

Convention:
  - 200 Hz substep stream (`_200hz.npz`) is the source of truth for foot
    kinematics, contact forces, and base state. Impact metrics and
    body acceleration use it.
  - 50 Hz control stream is used for tracking RMSEs and `action_rate_rms`
    (policy actions are predicted at 50 Hz).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


# Foot index in sim arrays: 0 = left, 1 = right.
FEET = {"left": 0, "right": 1}

# 12 substeps = 60 ms at 200 Hz. The fz peak can land several substeps after the touchdown firing instant.
_FZ_PEAK_WINDOW = 12

# Fall detection. Eval disables termination, so a fall is permanent: the robot stays
# collapsed for the rest of the window. Flag an env that ends it low and tilted.
# Healthy walking bottoms out ~0.70 m at < 0.15 tilt, so the margin is wide.
FALL_HEIGHT = 0.45  # base z [m]
FALL_TILT = 0.7  # |projected_gravity_xy|, ~44 deg lean
_FALL_TAIL_S = 0.5  # average the window tail so a noisy frame can't flip the verdict
_GRAVITY = np.array([0.0, 0.0, -1.0], dtype=np.float32)

# Every scenario commands 30 s of motion, so a walking robot steps continuously (~55
# touchdowns, both feet). Fewer than this is stumbling, not walking: its foot/effort
# metrics measure a couple of lucky steps, not a gait, so they don't count.
_MIN_STEPS = 10

# Honest diagnostics, kept on for every env: tracking error and step count get WORSE
# when the robot stands still (high error, ~0 steps), so they expose a frozen or
# barely-walking policy instead of hiding it. Every other metric is "lower is better"
# and gamed by not moving, so it is dropped for an env that didn't walk upright.
_ALWAYS_ON = {"vxy_rmse", "vyaw_rmse", "touchdown_count"}

# Baseline for the delta-vs-baseline heatmap. fastsac, "baseline_feetori" run.
# Source: https://wandb.ai/matteo-calabria01-maedmatt/IDSIA/runs/fvp6tbu7
# Measured on NASA (RTX 3070) at num_envs=64 with randomization:g1-benchmark, so the
# deltas are only meaningful for policies benched on the same hardware + config.
# NB: prior baseline was asad (RTX 4090); NASA numbers are NOT comparable to it.
# vyaw on non-yaw and vxy on yaw scenarios are NaN.
_BASELINE_NAME = "fastsac_baseline_feetori_fvp6tbu7"

_BASELINE_LOCO = {
    "forward_03": {"vxy_rmse": 0.1684, "vyaw_rmse": float("nan"), "action_rate_rms": 20.7537, "body_accel_rms": 3.8408},
    "forward_06": {"vxy_rmse": 0.1776, "vyaw_rmse": float("nan"), "action_rate_rms": 21.3398, "body_accel_rms": 4.2678},
    "forward_09": {"vxy_rmse": 0.2259, "vyaw_rmse": float("nan"), "action_rate_rms": 23.1862, "body_accel_rms": 4.9057},
    "backward_03": {"vxy_rmse": 0.1552, "vyaw_rmse": float("nan"), "action_rate_rms": 21.1619, "body_accel_rms": 3.4520},
    "backward_06": {"vxy_rmse": 0.2021, "vyaw_rmse": float("nan"), "action_rate_rms": 23.0143, "body_accel_rms": 3.5788},
    "backward_09": {"vxy_rmse": 0.2741, "vyaw_rmse": float("nan"), "action_rate_rms": 25.4203, "body_accel_rms": 4.1945},
    "strafe_left": {"vxy_rmse": 0.2125, "vyaw_rmse": float("nan"), "action_rate_rms": 20.8887, "body_accel_rms": 3.8606},
    "strafe_right": {"vxy_rmse": 0.1958, "vyaw_rmse": float("nan"), "action_rate_rms": 21.0663, "body_accel_rms": 3.8171},
    "yaw_left": {"vxy_rmse": float("nan"), "vyaw_rmse": 0.1539, "action_rate_rms": 20.7359, "body_accel_rms": 3.7589},
    "yaw_right": {"vxy_rmse": float("nan"), "vyaw_rmse": 0.1348, "action_rate_rms": 20.9989, "body_accel_rms": 3.9008},
    "avg": {"vxy_rmse": 0.2014, "vyaw_rmse": 0.1443, "action_rate_rms": 21.8566, "body_accel_rms": 3.9577},
}

_BASELINE_FEET = {
    "left": {
        "forward_03": {"touchdown_vz_peak": 0.8368, "touchdown_vz_rms": 0.8138, "fz_peak_at_touchdown_mean": 817.0503, "foot_accel_peak_at_touchdown_mean": 129.0774, "loading_rate_peak_at_touchdown_mean": 76703.8984, "foot_slip_speed_mean": 0.0109, "base_vz_peak_at_touchdown_mean": 0.3491, "base_accel_peak_at_touchdown_mean": 12.4458},
        "forward_06": {"touchdown_vz_peak": 0.8161, "touchdown_vz_rms": 0.7739, "fz_peak_at_touchdown_mean": 820.2939, "foot_accel_peak_at_touchdown_mean": 122.4063, "loading_rate_peak_at_touchdown_mean": 103569.5372, "foot_slip_speed_mean": 0.024, "base_vz_peak_at_touchdown_mean": 0.4412, "base_accel_peak_at_touchdown_mean": 12.9434},
        "forward_09": {"touchdown_vz_peak": 0.847, "touchdown_vz_rms": 0.7592, "fz_peak_at_touchdown_mean": 793.1153, "foot_accel_peak_at_touchdown_mean": 150.9989, "loading_rate_peak_at_touchdown_mean": 106066.2185, "foot_slip_speed_mean": 0.0388, "base_vz_peak_at_touchdown_mean": 0.537, "base_accel_peak_at_touchdown_mean": 12.538},
        "backward_03": {"touchdown_vz_peak": 0.8704, "touchdown_vz_rms": 0.8374, "fz_peak_at_touchdown_mean": 1180.6988, "foot_accel_peak_at_touchdown_mean": 127.735, "loading_rate_peak_at_touchdown_mean": 212913.5752, "foot_slip_speed_mean": 0.0117, "base_vz_peak_at_touchdown_mean": 0.2503, "base_accel_peak_at_touchdown_mean": 15.6325},
        "backward_06": {"touchdown_vz_peak": 0.9001, "touchdown_vz_rms": 0.8472, "fz_peak_at_touchdown_mean": 1156.6263, "foot_accel_peak_at_touchdown_mean": 139.3306, "loading_rate_peak_at_touchdown_mean": 203310.2683, "foot_slip_speed_mean": 0.0297, "base_vz_peak_at_touchdown_mean": 0.2882, "base_accel_peak_at_touchdown_mean": 13.1282},
        "backward_09": {"touchdown_vz_peak": 0.9795, "touchdown_vz_rms": 0.8958, "fz_peak_at_touchdown_mean": 1144.4947, "foot_accel_peak_at_touchdown_mean": 147.0062, "loading_rate_peak_at_touchdown_mean": 195815.6833, "foot_slip_speed_mean": 0.0591, "base_vz_peak_at_touchdown_mean": 0.3543, "base_accel_peak_at_touchdown_mean": 11.6772},
        "strafe_left": {"touchdown_vz_peak": 0.9021, "touchdown_vz_rms": 0.8819, "fz_peak_at_touchdown_mean": 1028.2224, "foot_accel_peak_at_touchdown_mean": 127.0523, "loading_rate_peak_at_touchdown_mean": 45025.3024, "foot_slip_speed_mean": 0.0095, "base_vz_peak_at_touchdown_mean": 0.3026, "base_accel_peak_at_touchdown_mean": 11.7652},
        "strafe_right": {"touchdown_vz_peak": 0.8867, "touchdown_vz_rms": 0.8508, "fz_peak_at_touchdown_mean": 1186.19, "foot_accel_peak_at_touchdown_mean": 109.1735, "loading_rate_peak_at_touchdown_mean": 198158.108, "foot_slip_speed_mean": 0.0109, "base_vz_peak_at_touchdown_mean": 0.2536, "base_accel_peak_at_touchdown_mean": 16.8851},
        "yaw_left": {"touchdown_vz_peak": 0.8953, "touchdown_vz_rms": 0.8757, "fz_peak_at_touchdown_mean": 1187.4948, "foot_accel_peak_at_touchdown_mean": 118.3354, "loading_rate_peak_at_touchdown_mean": 150300.0459, "foot_slip_speed_mean": 0.0071, "base_vz_peak_at_touchdown_mean": 0.3018, "base_accel_peak_at_touchdown_mean": 15.8356},
        "yaw_right": {"touchdown_vz_peak": 0.9167, "touchdown_vz_rms": 0.888, "fz_peak_at_touchdown_mean": 1229.302, "foot_accel_peak_at_touchdown_mean": 105.0751, "loading_rate_peak_at_touchdown_mean": 106064.3556, "foot_slip_speed_mean": 0.0082, "base_vz_peak_at_touchdown_mean": 0.2611, "base_accel_peak_at_touchdown_mean": 14.0761},
        "avg": {"touchdown_vz_peak": 0.8851, "touchdown_vz_rms": 0.8424, "fz_peak_at_touchdown_mean": 1054.3489, "foot_accel_peak_at_touchdown_mean": 127.6191, "loading_rate_peak_at_touchdown_mean": 139792.6993, "foot_slip_speed_mean": 0.021, "base_vz_peak_at_touchdown_mean": 0.3339, "base_accel_peak_at_touchdown_mean": 13.6927},
    },
    "right": {
        "forward_03": {"touchdown_vz_peak": 0.9626, "touchdown_vz_rms": 0.8995, "fz_peak_at_touchdown_mean": 931.1914, "foot_accel_peak_at_touchdown_mean": 151.8125, "loading_rate_peak_at_touchdown_mean": 85862.1607, "foot_slip_speed_mean": 0.0133, "base_vz_peak_at_touchdown_mean": 0.3952, "base_accel_peak_at_touchdown_mean": 14.3105},
        "forward_06": {"touchdown_vz_peak": 0.8955, "touchdown_vz_rms": 0.8129, "fz_peak_at_touchdown_mean": 880.7142, "foot_accel_peak_at_touchdown_mean": 125.4287, "loading_rate_peak_at_touchdown_mean": 116342.479, "foot_slip_speed_mean": 0.0276, "base_vz_peak_at_touchdown_mean": 0.4542, "base_accel_peak_at_touchdown_mean": 13.8217},
        "forward_09": {"touchdown_vz_peak": 0.9195, "touchdown_vz_rms": 0.7709, "fz_peak_at_touchdown_mean": 841.6933, "foot_accel_peak_at_touchdown_mean": 151.2883, "loading_rate_peak_at_touchdown_mean": 110781.1775, "foot_slip_speed_mean": 0.0486, "base_vz_peak_at_touchdown_mean": 0.537, "base_accel_peak_at_touchdown_mean": 12.6079},
        "backward_03": {"touchdown_vz_peak": 1.0508, "touchdown_vz_rms": 0.9871, "fz_peak_at_touchdown_mean": 1223.9215, "foot_accel_peak_at_touchdown_mean": 133.2084, "loading_rate_peak_at_touchdown_mean": 215462.0718, "foot_slip_speed_mean": 0.0134, "base_vz_peak_at_touchdown_mean": 0.322, "base_accel_peak_at_touchdown_mean": 16.081},
        "backward_06": {"touchdown_vz_peak": 1.0545, "touchdown_vz_rms": 1.0228, "fz_peak_at_touchdown_mean": 1242.6964, "foot_accel_peak_at_touchdown_mean": 156.8703, "loading_rate_peak_at_touchdown_mean": 215800.1802, "foot_slip_speed_mean": 0.0271, "base_vz_peak_at_touchdown_mean": 0.3488, "base_accel_peak_at_touchdown_mean": 13.7326},
        "backward_09": {"touchdown_vz_peak": 1.1182, "touchdown_vz_rms": 1.0585, "fz_peak_at_touchdown_mean": 1271.9137, "foot_accel_peak_at_touchdown_mean": 174.5385, "loading_rate_peak_at_touchdown_mean": 214755.3326, "foot_slip_speed_mean": 0.0531, "base_vz_peak_at_touchdown_mean": 0.3812, "base_accel_peak_at_touchdown_mean": 13.0923},
        "strafe_left": {"touchdown_vz_peak": 0.997, "touchdown_vz_rms": 0.9613, "fz_peak_at_touchdown_mean": 1324.9567, "foot_accel_peak_at_touchdown_mean": 127.1626, "loading_rate_peak_at_touchdown_mean": 226726.5435, "foot_slip_speed_mean": 0.0126, "base_vz_peak_at_touchdown_mean": 0.3273, "base_accel_peak_at_touchdown_mean": 19.228},
        "strafe_right": {"touchdown_vz_peak": 1.1526, "touchdown_vz_rms": 0.9846, "fz_peak_at_touchdown_mean": 1076.0105, "foot_accel_peak_at_touchdown_mean": 155.9937, "loading_rate_peak_at_touchdown_mean": 49129.9897, "foot_slip_speed_mean": 0.013, "base_vz_peak_at_touchdown_mean": 0.3659, "base_accel_peak_at_touchdown_mean": 12.1165},
        "yaw_left": {"touchdown_vz_peak": 0.9876, "touchdown_vz_rms": 0.9421, "fz_peak_at_touchdown_mean": 1174.3806, "foot_accel_peak_at_touchdown_mean": 123.4693, "loading_rate_peak_at_touchdown_mean": 53130.982, "foot_slip_speed_mean": 0.0131, "base_vz_peak_at_touchdown_mean": 0.2896, "base_accel_peak_at_touchdown_mean": 11.8742},
        "yaw_right": {"touchdown_vz_peak": 1.1062, "touchdown_vz_rms": 1.0214, "fz_peak_at_touchdown_mean": 1265.3617, "foot_accel_peak_at_touchdown_mean": 127.7695, "loading_rate_peak_at_touchdown_mean": 64694.6245, "foot_slip_speed_mean": 0.0113, "base_vz_peak_at_touchdown_mean": 0.3396, "base_accel_peak_at_touchdown_mean": 12.8062},
        "avg": {"touchdown_vz_peak": 1.0245, "touchdown_vz_rms": 0.9461, "fz_peak_at_touchdown_mean": 1123.284, "foot_accel_peak_at_touchdown_mean": 142.7542, "loading_rate_peak_at_touchdown_mean": 135268.5541, "foot_slip_speed_mean": 0.0233, "base_vz_peak_at_touchdown_mean": 0.3761, "base_accel_peak_at_touchdown_mean": 13.9671},
    },
}


def _mask_tracking_drift(row: dict, cmd: np.ndarray) -> None:
    """NaN-mask tracking metrics the scenario doesn't actively exercise.

    On a forward command, vyaw_rmse measures incidental yaw drift (e.g. from
    pitch coupling), not tracking quality. Small drifts on the unexercised
    axis vary a lot between policies and blow up percent deltas, drowning
    out the metrics that actually correspond to the scenario's task.
    """
    if np.max(np.abs(cmd[:, :2])) < 1e-3:
        row["vxy_rmse"] = float("nan")
    if np.max(np.abs(cmd[:, 2])) < 1e-3:
        row["vyaw_rmse"] = float("nan")


@dataclass
class Args:
    """Compute benchmark metrics from a single policy_eval recording."""

    policy_dir: Path
    """policy_eval/ directory to evaluate."""

    csv_out: Path | None = None
    """write per-scenario CSV here."""

    json_out: Path | None = None
    """write flat JSON here (wandb-ready)."""

    resume_from_ckpt: Path | None = None
    """checkpoint .pt that holds the wandb run path; resume that run and log delta-vs-baseline."""


def quat_rotate_inverse(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate world-frame vector into base frame; quaternion is (x, y, z, w).

    Mirrors utils/rotations.py:quat_rotate_inverse(w_last=True).
    """
    q_vec = quat_xyzw[..., :3]
    q_w = quat_xyzw[..., 3:4]
    a = vec * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, vec, axis=-1) * q_w * 2.0
    c = q_vec * np.sum(q_vec * vec, axis=-1, keepdims=True) * 2.0
    return a - b + c


def detect_fallen(npz: dict, tail_steps: int) -> np.ndarray:
    """Per-env boolean: did the env end the active window low and tilted?"""
    active = npz["is_active"]
    base_pos = npz["base_pos"][active]  # (T_active, num_envs, 3)
    base_quat = npz["base_quat"][active]  # (T_active, num_envs, 4)
    grav = np.broadcast_to(_GRAVITY, base_pos.reshape(-1, 3).shape)
    grav_base = quat_rotate_inverse(base_quat.reshape(-1, 4), grav).reshape(base_pos.shape)
    tilt = np.linalg.norm(grav_base[..., :2], axis=-1)  # (T_active, num_envs)
    tail = max(1, tail_steps)
    height_end = base_pos[-tail:, :, 2].mean(axis=0)
    tilt_end = tilt[-tail:].mean(axis=0)
    return (height_end < FALL_HEIGHT) & (tilt_end > FALL_TILT)


def detect_touchdowns(vz: np.ndarray, fz: np.ndarray,
                      arm_vz: float = -0.3, fire_fz: float = 1.0) -> np.ndarray:
    """Touchdown indices for one foot.

    Arm when vz < arm_vz; fire on rising edge of fz through fire_fz.
    Same algorithm as utils/touchdown.py:TouchdownDetector.
    """
    events = []
    armed = False
    for i in range(len(vz)):
        if not armed and vz[i] < arm_vz:
            armed = True
        if armed and i > 0 and fz[i - 1] < fire_fz <= fz[i]:
            events.append(i)
            armed = False
    return np.asarray(events, dtype=np.int64)


def detect_walked(npz_hi: dict, num_envs: int) -> np.ndarray:
    """Per-env boolean: did the robot step at a walking cadence in the active window?"""
    active = npz_hi["is_active"]
    walked = np.zeros(num_envs, dtype=bool)
    for n in range(num_envs):
        steps = 0
        for foot in FEET.values():
            td = detect_touchdowns(npz_hi["foot_vel"][:, n, foot, 2], npz_hi["foot_force"][:, n, foot, 2])
            steps += int(np.count_nonzero(active[td]))
        walked[n] = steps >= _MIN_STEPS
    return walked


def vxy_rmse(vel_body_xy: np.ndarray, cmd_xy: np.ndarray) -> float:
    """Body-frame linear velocity tracking RMSE.

    Reference: tracking_lin_vel reward, legged_gym / walk-these-ways
    (Margolis & Agrawal, CoRL 2023, arXiv:2212.03238).
    """
    err = vel_body_xy - cmd_xy
    return float(np.sqrt(np.mean(np.sum(err**2, axis=-1))))


def vyaw_rmse(omega_body_z: np.ndarray, cmd_yaw: np.ndarray) -> float:
    """Yaw rate tracking RMSE.

    Reference: tracking_ang_vel reward, legged_gym.
    """
    return float(np.sqrt(np.mean((omega_body_z - cmd_yaw) ** 2)))


def touchdown_vz_peak(vz_at_td: np.ndarray) -> float:
    """Peak downward foot velocity across touchdown events.

    Reference: walk-these-ways _reward_feet_impact_vel
    (Margolis & Agrawal, arXiv:2212.03238).
    """
    if len(vz_at_td) == 0:
        return 0.0
    return float(np.max(np.abs(np.minimum(vz_at_td, 0.0))))


def touchdown_vz_rms(vz_at_td: np.ndarray) -> float:
    """RMS downward foot velocity across touchdown events.

    Reference: walk-these-ways _reward_feet_impact_vel
    (Margolis & Agrawal, arXiv:2212.03238).
    """
    if len(vz_at_td) == 0:
        return 0.0
    vz_down = np.minimum(vz_at_td, 0.0)
    return float(np.sqrt(np.mean(vz_down**2)))


def _peak_in_window(sig: np.ndarray, td_idx: np.ndarray) -> float:
    """Mean over touchdowns of the peak of `sig` in the [i, i+window) impact window.

    The peak lands a few substeps after the firing instant, so window then max.
    Used for every per-impact peak: fz (QuietWalk, arXiv:2604.23702), tangential
    force, foot accel, loading rate. Skips events whose window starts past the end
    (finite-diff signals are one sample shorter than the force/velocity traces).
    """
    peaks = [sig[i : i + _FZ_PEAK_WINDOW].max() for i in td_idx if i < len(sig)]
    return float(np.mean(peaks)) if peaks else 0.0


def action_rate_rms(action: np.ndarray, dt: float) -> float:
    """RMS action first-difference per second.

    Reference: action_rate_l2 reward, legged_gym.
    """
    da_per_s = np.diff(action, axis=0) / dt
    return float(np.sqrt(np.mean(np.sum(da_per_s**2, axis=-1))))


def body_accel_rms(lin_vel_world: np.ndarray, dt: float) -> float:
    """RMS base linear acceleration magnitude (finite-diff of base velocity)."""
    a = np.diff(lin_vel_world, axis=0) / dt
    return float(np.sqrt(np.mean(np.sum(a**2, axis=-1))))


def _foot_metrics(npz_hi: dict, foot_idx: int, env_idx: int, sim_dt: float) -> dict[str, float]:
    """Touchdown-derived metrics for one foot on one env. Events outside is_active are dropped."""
    vel = npz_hi["foot_vel"][:, env_idx, foot_idx, :]      # (T, 3) world-frame foot velocity
    force = npz_hi["foot_force"][:, env_idx, foot_idx, :]   # (T, 3) contact force
    vz = vel[:, 2]
    fz = force[:, 2]
    td = detect_touchdowns(vz, fz)
    td = td[npz_hi["is_active"][td]]
    # Approach velocity spans the threshold crossing: take the more downward of
    # {pre, post}; whichever is pre-impulse holds the actual impact speed.
    vz_at_td = np.minimum(vz[td - 1], vz[td])
    # Finite-diff at 200 Hz: the impact shock and force loading rate are sub-20ms, so the
    # 50 Hz control stream aliases them away. diff() returns T-1 samples (_peak_in_window guards).
    foot_accel = np.linalg.norm(np.diff(vel, axis=0) / sim_dt, axis=-1)
    loading_rate = np.diff(fz) / sim_dt
    # Slip/scuff: world-frame horizontal foot speed while loaded. ~0 for a planted foot,
    # >0 only when it slides -- the channel the (normal-only) contact force can't see.
    in_contact = fz > 1.0  # matches detect_touchdowns fire_fz
    foot_horiz_speed = np.linalg.norm(vel[:, :2], axis=-1)
    slip_speed = float(np.mean(foot_horiz_speed[in_contact])) if in_contact.any() else 0.0
    # Base-drop channel: a soft-looking foot landing can hide the impact in the body dropping
    # with it. The foot metrics above are blind to this; the base vertical shock is not.
    base_vz = npz_hi["base_lin_vel"][:, env_idx, 2]
    base_vz_peak = _peak_in_window(np.maximum(-base_vz, 0.0), td)
    base_accel_peak = _peak_in_window(np.abs(np.diff(base_vz) / sim_dt), td)
    return {
        "touchdown_count": float(len(td)),
        "touchdown_vz_peak": touchdown_vz_peak(vz_at_td),
        "touchdown_vz_rms": touchdown_vz_rms(vz_at_td),
        "fz_peak_at_touchdown_mean": _peak_in_window(fz, td),
        "foot_accel_peak_at_touchdown_mean": _peak_in_window(foot_accel, td),
        "loading_rate_peak_at_touchdown_mean": _peak_in_window(loading_rate, td),
        "foot_slip_speed_mean": slip_speed,
        "base_vz_peak_at_touchdown_mean": base_vz_peak,
        "base_accel_peak_at_touchdown_mean": base_accel_peak,
    }


def _body_metrics(npz: dict, npz_hi: dict, dt: float, sim_dt: float, env_idx: int) -> dict[str, float]:
    """Locomotion-quality metrics for one env: tracking + smoothness. Foot-agnostic."""
    active = npz["is_active"]
    active_hi = npz_hi["is_active"]
    vel_body = quat_rotate_inverse(npz["base_quat"][active, env_idx], npz["base_lin_vel"][active, env_idx])
    omega_body = quat_rotate_inverse(npz["base_quat"][active, env_idx], npz["base_ang_vel"][active, env_idx])
    return {
        "vxy_rmse": vxy_rmse(vel_body[:, :2], npz["cmd"][active, env_idx, :2]),
        "vyaw_rmse": vyaw_rmse(omega_body[:, 2], npz["cmd"][active, env_idx, 2]),
        "action_rate_rms": action_rate_rms(npz["action"][active, env_idx], dt),
        "body_accel_rms": body_accel_rms(npz_hi["base_lin_vel"][active_hi, env_idx], sim_dt),
    }


def _aggregate(per_env: list[dict]) -> tuple[dict, dict]:
    """NaN-aware element-wise mean and std across a list of per-env metric dicts."""
    keys = list(per_env[0].keys())
    mean: dict = {}
    std: dict = {}
    for k in keys:
        vals = np.array([r[k] for r in per_env], dtype=float)
        finite = vals[~np.isnan(vals)]
        mean[k] = float(np.mean(finite)) if finite.size else float("nan")
        std[k] = float(np.std(finite)) if finite.size else float("nan")
    return mean, std


def _append_avg(rows: list[dict]) -> list[str]:
    """Append the 'avg' row in place (mean over non-NaN cells per metric). Returns metric names."""
    metric_names = [k for k in rows[0] if k != "scenario"]
    avg: dict = {"scenario": "avg"}
    for m in metric_names:
        vals = [r[m] for r in rows if not np.isnan(r[m])]
        avg[m] = float(np.mean(vals)) if vals else float("nan")
    rows.append(avg)
    return metric_names


def load_policy_dir(policy_dir: Path) -> tuple[dict, list[tuple[str, dict, dict]]]:
    """Load manifest and (50 Hz, 200 Hz) NPZ pairs from a policy_eval directory."""
    manifest = json.loads((policy_dir / "manifest.json").read_text())
    scenarios = []
    for entry in manifest["scenarios"]:
        stem = f"scenario_{entry['index']:02d}_{entry['name']}"
        lo = dict(np.load(policy_dir / f"{stem}.npz"))
        hi = dict(np.load(policy_dir / f"{stem}_200hz.npz"))
        scenarios.append((entry["name"], lo, hi))
    return manifest, scenarios


def render_markdown(rows_mean: list[dict], rows_std: list[dict], metric_names: list[str]) -> str:
    """Format each cell as `mean ± std`. NaN means propagate as 'nan'."""
    headers = ["scenario"] + metric_names
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for rm, rs in zip(rows_mean, rows_std):
        cells = [rm["scenario"]]
        for m in metric_names:
            mean = rm[m]
            if m.endswith("_rate"):
                cells.append(f"{mean:.2f}")  # failure fraction (fall/nowalk); no ±std
            elif np.isnan(mean):
                cells.append("nan")
            else:
                cells.append(f"{mean:.4f}±{rs[m]:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _delta_matrix(rows: list[dict], metrics: list[str], baseline: dict) -> np.ndarray:
    """Percent delta of each (scenario, metric) cell vs `baseline`. NaN where masked."""
    deltas = np.empty((len(rows), len(metrics)))
    for i, r in enumerate(rows):
        for j, m in enumerate(metrics):
            if m.endswith("_rate"):
                # Failure rates have no baseline; show the absolute percent (0 = green,
                # higher = redder), consistent with "positive delta = worse".
                deltas[i, j] = r[m] * 100.0
                continue
            base = baseline[r["scenario"]][m]
            deltas[i, j] = (r[m] - base) / base * 100.0  # NaN-in (masked) propagates
    return deltas


def _heatmap(deltas: np.ndarray, metrics: list[str], scens: list[str],
             title: str, vbound: float):
    """Plotly delta-vs-baseline heatmap with caller-supplied symmetric color bound."""
    import plotly.express as px
    return px.imshow(
        deltas, x=metrics, y=scens,
        color_continuous_scale="RdYlGn_r",
        zmin=-vbound, zmax=vbound,
        text_auto=".1f",
        aspect="auto",
        labels={"color": "% Δ"},
        title=f"{title}: % delta vs {_BASELINE_NAME} (negative = better)",
    )


def _log_to_wandb(loco_rows: list[dict], loco_metrics: list[str],
                  per_leg: dict[str, tuple[list[dict], list[str]]],
                  ckpt_path: Path, policy_dir: Path) -> None:
    """Resume the wandb run from the checkpoint metadata; log locomotion + per-leg sound heatmaps."""
    import torch
    import wandb

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "wandb_run_path" not in state:
        raise ValueError(f"{ckpt_path} has no wandb_run_path metadata.")
    entity, project, run_id = state["wandb_run_path"].split("/")
    # dir= keeps the resumed run's staging files under the eval dir, not cwd.
    # x_disable_meta=True prevents the resume from overwriting training's argv/program.
    wandb.init(
        entity=entity, project=project, id=run_id, resume="allow",
        dir=policy_dir, settings=wandb.Settings(x_disable_meta=True),
    )

    # Locomotion heatmap: own color bound (metric domain unrelated to sound).
    # The *_rate columns are absolute percents, not deltas; keep them out of the bound
    # so a failed run's red column doesn't flatten the quality deltas (it just clips).
    loco_deltas = _delta_matrix(loco_rows, loco_metrics, _BASELINE_LOCO)
    quality = np.array([not m.endswith("_rate") for m in loco_metrics])
    loco_vbound = float(np.nanmax(np.abs(loco_deltas[:, quality])))
    wandb.log({"bench/delta_heatmap_locomotion": _heatmap(
        loco_deltas, loco_metrics, [r["scenario"] for r in loco_rows],
        title="Locomotion quality", vbound=loco_vbound,
    )})

    # Sound heatmaps: touchdown_count is a sanity-check diagnostic with no baseline,
    # drop it. Shared color bound across L/R so asymmetries are visually comparable.
    sound = []
    for leg, (rows, metric_names) in per_leg.items():
        # Heatmap needs a baseline scalar per metric. New metrics still show in the
        # markdown/CSV/JSON tables but are excluded here until _BASELINE_FEET is regenerated.
        base = _BASELINE_FEET[leg]["avg"]
        metrics = [m for m in metric_names if m in base and not m.startswith("touchdown_count")]
        sound.append((leg, rows, metrics, _delta_matrix(rows, metrics, _BASELINE_FEET[leg])))
    sound_vbound = float(np.nanmax(np.abs(np.concatenate([d.flatten() for _, _, _, d in sound]))))
    for leg, rows, metrics, deltas in sound:
        wandb.log({f"bench/delta_heatmap_sound_{leg}": _heatmap(
            deltas, metrics, [r["scenario"] for r in rows],
            title=f"Sound quality, {leg} foot", vbound=sound_vbound,
        )})

    # Pin the exact ckpt used so the bench result is reproducible from the wandb run page.
    wandb.run.summary["bench/checkpoint"] = str(ckpt_path)
    wandb.finish()


def main(args: Args) -> None:
    manifest, scenarios = load_policy_dir(args.policy_dir)
    dt = 1.0 / manifest["policy_hz"]
    sim_dt = 1.0 / manifest["sim_hz"]
    # Manifest field added with multi-env benchmarking; old recordings detect from shape.
    num_envs = int(manifest.get("num_envs", scenarios[0][1]["cmd"].shape[1]))

    print(f"# {args.policy_dir.name}  (num_envs={num_envs})")

    per_env_dump: dict[str, dict] = {}
    tail_steps = int(round(_FALL_TAIL_S / dt))
    # An env's metrics count only if it did the task: stayed upright AND actually
    # stepped. One that fell or froze games every "lower is better" metric, so it is
    # dropped from all of them; fall_rate / nowalk_rate report how many failed, and how.
    fallen_by_scenario = {name: detect_fallen(lo, tail_steps) for name, lo, _ in scenarios}
    walked_by_scenario = {name: detect_walked(hi, num_envs) for name, _, hi in scenarios}

    # Locomotion quality: tracking + smoothness, shared across legs.
    loco_mean = []
    loco_std = []
    for name, lo, hi in scenarios:
        # cmd is broadcast across envs, so masking decision is identical across envs.
        cmd_for_mask = lo["cmd"][:, 0]
        fallen, walked = fallen_by_scenario[name], walked_by_scenario[name]
        valid = ~fallen & walked
        per_env = []
        for n in range(num_envs):
            row = _body_metrics(lo, hi, dt, sim_dt, env_idx=n)
            _mask_tracking_drift(row, cmd_for_mask)
            if not valid[n]:  # failed the task: keep only the honest diagnostics
                row = {k: (v if k in _ALWAYS_ON else float("nan")) for k, v in row.items()}
            per_env.append(row)
        mean, std = _aggregate(per_env)
        mean["fall_rate"] = float(fallen.mean())
        mean["nowalk_rate"] = float((~fallen & ~walked).mean())
        std["fall_rate"] = std["nowalk_rate"] = float("nan")
        loco_mean.append({"scenario": name, **mean})
        loco_std.append({"scenario": name, **std})
        per_env_dump.setdefault(name, {})["locomotion"] = per_env
    loco_metrics = _append_avg(loco_mean)
    _append_avg(loco_std)
    print("\n## Locomotion quality")
    print(render_markdown(loco_mean, loco_std, loco_metrics))

    # Sound quality: per-leg foot impact metrics. Asymmetry between L/R is the signal.
    per_leg_mean: dict[str, list[dict]] = {}
    per_leg_std: dict[str, list[dict]] = {}
    per_leg_metrics: dict[str, list[str]] = {}
    for leg, idx in FEET.items():
        mean_rows = []
        std_rows = []
        for name, _, hi in scenarios:
            fallen, walked = fallen_by_scenario[name], walked_by_scenario[name]
            valid = ~fallen & walked
            per_env = []
            for n in range(num_envs):
                row = _foot_metrics(hi, idx, env_idx=n, sim_dt=sim_dt)
                if not valid[n]:  # failed the task: keep only the honest diagnostics
                    row = {k: (v if k in _ALWAYS_ON else float("nan")) for k, v in row.items()}
                per_env.append(row)
            mean, std = _aggregate(per_env)
            mean_rows.append({"scenario": name, **mean})
            std_rows.append({"scenario": name, **std})
            per_env_dump.setdefault(name, {}).setdefault("feet", {})[leg] = per_env
        metrics = _append_avg(mean_rows)
        _append_avg(std_rows)
        per_leg_mean[leg] = mean_rows
        per_leg_std[leg] = std_rows
        per_leg_metrics[leg] = metrics
        print(f"\n## Sound quality, {leg} foot")
        print(render_markdown(mean_rows, std_rows, metrics))

    per_env_path = args.policy_dir / "per_env_metrics.json"
    per_env_path.write_text(json.dumps(per_env_dump, indent=2))
    print(f"\nwrote {per_env_path}")

    if args.csv_out:
        sound_metrics = per_leg_metrics["left"]
        cols = (
            [f"{m}_mean" for m in loco_metrics] + [f"{m}_std" for m in loco_metrics]
            + [f"{m}_mean" for m in sound_metrics] + [f"{m}_std" for m in sound_metrics]
        )
        with args.csv_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["section", "scenario"] + cols, extrasaction="ignore")
            w.writeheader()
            for rm, rs in zip(loco_mean, loco_std):
                row = {"section": "locomotion", "scenario": rm["scenario"]}
                for m in loco_metrics:
                    row[f"{m}_mean"] = rm[m]
                    row[f"{m}_std"] = rs[m]
                w.writerow(row)
            for leg in FEET:
                for rm, rs in zip(per_leg_mean[leg], per_leg_std[leg]):
                    row = {"section": f"sound_{leg}", "scenario": rm["scenario"]}
                    for m in per_leg_metrics[leg]:
                        row[f"{m}_mean"] = rm[m]
                        row[f"{m}_std"] = rs[m]
                    w.writerow(row)
        print(f"wrote {args.csv_out}")

    if args.json_out:
        flat = {}
        for rm, rs in zip(loco_mean, loco_std):
            for m in loco_metrics:
                flat[f"locomotion/{rm['scenario']}/{m}/mean"] = rm[m]
                flat[f"locomotion/{rm['scenario']}/{m}/std"] = rs[m]
        for leg in FEET:
            for rm, rs in zip(per_leg_mean[leg], per_leg_std[leg]):
                for m in per_leg_metrics[leg]:
                    flat[f"sound_{leg}/{rm['scenario']}/{m}/mean"] = rm[m]
                    flat[f"sound_{leg}/{rm['scenario']}/{m}/std"] = rs[m]
        args.json_out.write_text(json.dumps(flat, indent=2))
        print(f"wrote {args.json_out}")

    if args.resume_from_ckpt:
        # Heatmap compares mean against the (single-env) baseline scalars. Std is logged
        # to the per-env file only; rerun the baseline under the new bench config to
        # refresh _BASELINE before trusting heatmap deltas.
        per_leg_for_wandb = {leg: (per_leg_mean[leg], per_leg_metrics[leg]) for leg in FEET}
        _log_to_wandb(loco_mean, loco_metrics, per_leg_for_wandb, args.resume_from_ckpt, args.policy_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
