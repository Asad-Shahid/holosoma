"""Per-foot contact and touchdown statistics, computed once per control step.

Owns the model-free foot signals every consumer shares: contact state and
duration, touchdown events (via TouchdownDetector), the foot velocity sampled
just before impact, and a short post-touchdown force window. The locomotion
env steps, resets, and logs this object under ``feet/``; reward terms and
callbacks read its public tensors instead of tracking their own copies.
"""

from __future__ import annotations

import torch
from torch import Tensor

from holosoma.utils.touchdown import TouchdownDetector


class FootState:
    """Stateful per-foot signals over [num_envs, num_feet].

    After each step() the public tensors describe the current control step:

    - fired: touchdown fired this step (bool)
    - vz_pre: vertical foot velocity one step before this one
    - approach_speed: downward component of vz_pre (>= 0)
    - fz: vertical contact force this step
    - contact: foot in contact this step (fz > contact_force_threshold)
    - contact_duration_s: running duration of the current contact
    """

    def __init__(
        self,
        num_envs: int,
        num_feet: int,
        dt: float,
        device: torch.device | str,
        *,
        vz_arm_thr: float = 0.3,
        fz_fire_thr: float = 1.0,
        contact_force_threshold: float = 1.0,
        force_window_s: float = 0.06,
    ) -> None:
        self._dt = dt
        self._contact_force_threshold = contact_force_threshold
        self._detector = TouchdownDetector(
            num_envs=num_envs, num_feet=num_feet, vz_arm_thr=vz_arm_thr, fz_fire_thr=fz_fire_thr
        )
        self._window_steps = max(1, round(force_window_s / dt))
        self._window_label = f"{round(force_window_s * 1000)}ms"

        shape = (num_envs, num_feet)
        self.fired = torch.zeros(shape, dtype=torch.bool, device=device)
        self.vz_pre = torch.zeros(shape, device=device)
        self.approach_speed = torch.zeros(shape, device=device)
        self.fz = torch.zeros(shape, device=device)
        self.contact = torch.zeros(shape, dtype=torch.bool, device=device)
        self.contact_duration_s = torch.zeros(shape, device=device)

        self._vz_prev = torch.zeros(shape, device=device)
        self._last_contact = torch.zeros(shape, dtype=torch.bool, device=device)
        self._liftoff_duration_s = torch.full(shape, float("nan"), device=device)
        self._window_active = torch.zeros(shape, dtype=torch.bool, device=device)
        self._window_remaining = torch.zeros(shape, dtype=torch.long, device=device)
        self._window_initial_fz = torch.zeros(shape, device=device)
        self._window_peak_fz = torch.zeros(shape, device=device)
        self._window_impulse = torch.zeros(shape, device=device)
        self._window_completed = torch.zeros(shape, dtype=torch.bool, device=device)

    def step(self, vz: Tensor, fz: Tensor) -> None:
        """Advance one control step from per-foot vertical velocity and contact force."""
        self.fz = fz
        self.fired = self._detector.step(vz, fz)
        self.vz_pre = self._vz_prev
        self.approach_speed = (-self.vz_pre).clamp(min=0.0)

        contact = fz > self._contact_force_threshold
        step_s = torch.full_like(self.contact_duration_s, self._dt)
        duration = torch.where(
            contact,
            torch.where(self._last_contact, self.contact_duration_s + step_s, step_s),
            torch.zeros_like(self.contact_duration_s),
        )
        ended = (~contact) & self._last_contact
        nan = torch.full_like(fz, float("nan"))
        self._liftoff_duration_s = torch.where(ended, self.contact_duration_s, nan)
        self.contact = contact
        self.contact_duration_s = duration
        self._last_contact = contact

        # Post-touchdown force window: opened on fire, accumulates peak/impulse,
        # closes after _window_steps. A new fire on the same foot restarts it.
        new_window = torch.full_like(self._window_remaining, self._window_steps)
        self._window_active = self._window_active | self.fired
        self._window_remaining = torch.where(self.fired, new_window, self._window_remaining)
        self._window_initial_fz = torch.where(self.fired, fz, self._window_initial_fz)
        self._window_peak_fz = torch.where(self.fired, fz, self._window_peak_fz)
        self._window_impulse = torch.where(self.fired, torch.zeros_like(self._window_impulse), self._window_impulse)

        active = self._window_active
        self._window_peak_fz = torch.where(active, torch.maximum(self._window_peak_fz, fz), self._window_peak_fz)
        self._window_impulse = torch.where(active, self._window_impulse + fz * self._dt, self._window_impulse)
        remaining = torch.where(active, self._window_remaining - 1, self._window_remaining)
        self._window_completed = active & (remaining <= 0)
        self._window_remaining = torch.where(self._window_completed, torch.zeros_like(remaining), remaining)
        self._window_active = active & ~self._window_completed

        self._vz_prev = vz

    def log(
        self,
        log_dict: dict,
        phase: torch.Tensor | None = None,
        foot_height: torch.Tensor | None = None,
    ) -> None:
        """Write the feet/ metrics for the current step into log_dict.

        Event timeline per touchdown: pre_touchdown_vz is sampled one step
        before the fire (last free-flight velocity), touchdown_fz at the fire
        step, and the fz_peak/impulse/ratio over the 60ms window after it.

        phase / foot_height (per-foot, fed by the env which owns those signals)
        are logged at the fire step as a gait-gate check.
        """
        nan = torch.full_like(self.fz, float("nan"))
        label = self._window_label
        log_dict["feet/touchdown_rate"] = self.fired.any(dim=1).float().mean()
        vz_events = torch.where(self.fired, self.vz_pre, nan)
        vz_mean = vz_events.nanmean()
        log_dict["feet/pre_touchdown_vz"] = vz_mean
        # spread across touchdowns: the slam tail can worsen while the mean improves
        log_dict["feet/pre_touchdown_vz_std"] = (vz_events - vz_mean).square().nanmean().sqrt()
        log_dict["feet/touchdown_fz"] = torch.where(self.fired, self.fz, nan).nanmean()
        log_dict["feet/contact_rate"] = self.contact.float().mean()
        log_dict["feet/stance_time_s"] = self._liftoff_duration_s.nanmean()
        peak_to_initial = self._window_peak_fz / self._window_initial_fz.clamp(min=1e-3)
        log_dict[f"feet/fz_peak_{label}"] = torch.where(self._window_completed, self._window_peak_fz, nan).nanmean()
        log_dict[f"feet/impulse_{label}"] = torch.where(self._window_completed, self._window_impulse, nan).nanmean()
        log_dict[f"feet/fz_peak_ratio_{label}"] = torch.where(self._window_completed, peak_to_initial, nan).nanmean()
        # Gait-gate check: where in the cycle / how high the foot is at touchdown.
        # Exposes a misaligned phase/height gate.
        if phase is not None:
            phase_progress = torch.remainder((phase + torch.pi) / (2 * torch.pi), 1.0)
            log_dict["feet/phase_at_touchdown"] = torch.where(self.fired, phase_progress, nan).nanmean()
        if foot_height is not None:
            log_dict["feet/foot_height_at_touchdown"] = torch.where(self.fired, foot_height, nan).nanmean()
        if self.fz.shape[1] >= 2:
            left, right = self.fired[:, 0], self.fired[:, 1]
            log_dict["feet/pre_touchdown_vz_left"] = torch.where(left, self.vz_pre[:, 0], nan[:, 0]).nanmean()
            log_dict["feet/pre_touchdown_vz_right"] = torch.where(right, self.vz_pre[:, 1], nan[:, 1]).nanmean()
            log_dict["feet/touchdown_fz_left"] = torch.where(left, self.fz[:, 0], nan[:, 0]).nanmean()
            log_dict["feet/touchdown_fz_right"] = torch.where(right, self.fz[:, 1], nan[:, 1]).nanmean()

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Clear per-foot state. None = all envs; tensor = those envs only."""
        self._detector.reset(env_ids)
        if env_ids is None:
            for t in (
                self.vz_pre,
                self.approach_speed,
                self.fz,
                self.contact_duration_s,
                self._vz_prev,
                self._window_remaining,
                self._window_initial_fz,
                self._window_peak_fz,
                self._window_impulse,
            ):
                t.zero_()
            for b in (self.fired, self.contact, self._last_contact, self._window_active, self._window_completed):
                b.zero_()
            self._liftoff_duration_s.fill_(float("nan"))
        else:
            for t in (
                self.vz_pre,
                self.approach_speed,
                self.fz,
                self.contact_duration_s,
                self._vz_prev,
                self._window_remaining,
                self._window_initial_fz,
                self._window_peak_fz,
                self._window_impulse,
            ):
                t[env_ids] = 0
            for b in (self.fired, self.contact, self._last_contact, self._window_active, self._window_completed):
                b[env_ids] = False
            self._liftoff_duration_s[env_ids] = float("nan")
