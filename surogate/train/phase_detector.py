"""Training-phase detection from per-update losses (and, optionally, gradient norms).

Phases
------
WARMUP      First ``warmup`` updates: statistics are unreliable.
CONVERGING  The loss is decreasing by more than its own noise.
PLATEAU     The loss change is within noise.
UNSTABLE    A genuine instability signature (see below).
DIVERGING   The loss is rising by more than its own noise and by a material amount.

Every test is noise-aware: it compares a statistic of the recent updates with
the same statistic of the preceding updates, scaled by the run's own noise.
The absolute per-update noise level (coefficient of variation) never triggers
a phase by itself: a loss averaged over a few dozen mixed-difficulty examples
legitimately has a per-update CV of 0.3-0.5 for an entire healthy run.

UNSTABLE fires when any of these hold (rule 1 always, the others after warm-up):

1. non-finite  the loss or the gradient norm is NaN/inf (immediately);
2. dispersion  the winsorized std of the detrended recent half of the window
               exceeds the older half's by ``dispersion_ratio``;
3. spikes      at least ``spike_count`` of the last ``spike_span`` losses lie
               above the older-half mean + ``spike_sigmas`` robust sigmas;
4. gradients   (only when ``grad_norm`` is passed) the median norm of the
               recent half exceeds the older half's by ``norm_ratio``, or the
               fraction of updates at/above ``clip_threshold`` rises by
               ``clip_fraction_rise`` between the halves.

Isolated single spikes do not trigger: medians, the MAD and the winsorized std
ignore them, and a spike cluster needs ``spike_count`` hits.

The trend is judged over ``trend_window`` updates (default ``2 * window``) with
a two-sample z-score of the half means whose standard error uses the pooled
within-half detrended residual std, so a steady slope is not counted as noise:

* z >= ``diverge_z`` and relative rise >= ``diverge_threshold``   -> DIVERGING
* -z >= ``converge_z`` and relative drop >= ``plateau_threshold`` -> CONVERGING
  (hysteresis: CONVERGING persists until -z < ``converge_exit_z``)
* otherwise                                                        -> PLATEAU

UNSTABLE and DIVERGING are held for ``hold_steps`` updates so a transient
cannot flap the phase.  Only phase changes are logged; ``diagnostics()``
returns the statistics behind the latest decision.
"""

import math
from collections import deque
from enum import Enum

import numpy as np


class TrainingPhase(Enum):
    WARMUP = "warmup"
    CONVERGING = "converging"
    PLATEAU = "plateau"
    UNSTABLE = "unstable"
    DIVERGING = "diverging"


_MAD_TO_SIGMA = 1.4826
_SEVERITY = {TrainingPhase.UNSTABLE: 2, TrainingPhase.DIVERGING: 1}
_STAT_KEYS = (
    "n_window",
    "n_norm",
    "n_trend",
    "older_mean",
    "recent_mean",
    "rel_change",
    "z_shift",
    "sigma_pooled",
    "se_diff",
    "sigma_older",
    "sigma_recent",
    "disp_ratio",
    "sigma_robust",
    "spike_level",
    "spike_count",
    "norm_median_older",
    "norm_median_recent",
    "norm_ratio",
    "clip_frac_older",
    "clip_frac_recent",
    "clip_frac_rise",
)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _detrend(values: np.ndarray) -> np.ndarray:
    """Residuals of a least-squares line fitted to *values* (index as x)."""
    n = values.size
    if n < 3:
        return values - values.mean() if n else values
    x = np.arange(n, dtype=np.float64)
    x -= x.mean()
    centered = values - values.mean()
    slope = float(np.dot(x, centered)) / max(float(np.dot(x, x)), 1e-12)
    return centered - slope * x


def _robust_sigma(residuals: np.ndarray) -> float:
    """Median-absolute-deviation estimate of the noise sigma (0 for constant data)."""
    if residuals.size == 0:
        return 0.0
    return _MAD_TO_SIGMA * float(np.median(np.abs(residuals - np.median(residuals))))


def _winsorized_sigma(residuals: np.ndarray, clip_sigmas: float = 5.0) -> float:
    """Std of residuals clipped at +/- ``clip_sigmas`` robust sigmas (ddof=2 for the line).

    Isolated spikes are clipped away, but a genuine widening of the bulk of the
    noise is tracked; the MAD alone is noisier and reacts to shape changes.
    """
    n = residuals.size
    if n < 3:
        return 0.0
    robust = _robust_sigma(residuals)
    if robust > 0:
        center = float(np.median(residuals))
        residuals = np.clip(residuals, center - clip_sigmas * robust, center + clip_sigmas * robust)
    return float(np.sqrt(np.sum((residuals - residuals.mean()) ** 2) / (n - 2)))


def _ratio(numerator: float, denominator: float) -> float:
    if denominator > 0:
        return numerator / denominator
    return math.inf if numerator > 0 else 1.0


class PhaseDetector:
    """Classifies the current training phase and logs transitions.

    ``step(loss, step, grad_norm=None)`` keeps the historical two-argument call
    working; passing the update's gradient norm enables the gradient rules.
    """

    def __init__(
        self,
        logger,
        window: int = 100,
        warmup: int = 50,
        plateau_threshold: float = 0.001,
        diverge_threshold: float = 0.05,
        trend_window: int | None = None,
        converge_z: float = 2.0,
        converge_exit_z: float = 1.0,
        diverge_z: float = 3.5,
        dispersion_ratio: float = 2.5,
        spike_sigmas: float = 4.0,
        spike_count: int = 3,
        spike_span: int = 10,
        norm_ratio: float = 3.0,
        clip_threshold: float = 1.0,
        clip_fraction_rise: float = 0.5,
        min_half: int = 25,
        hold_steps: int = 20,
    ):
        self.logger = logger
        self.window = int(window)
        self.trend_window = int(trend_window) if trend_window else 2 * self.window
        self.warmup = int(warmup)
        self.plateau_threshold = float(plateau_threshold)
        self.diverge_threshold = float(diverge_threshold)
        self.converge_z = float(converge_z)
        self.converge_exit_z = float(converge_exit_z)
        self.diverge_z = float(diverge_z)
        self.dispersion_ratio = float(dispersion_ratio)
        self.spike_sigmas = float(spike_sigmas)
        self.spike_count = int(spike_count)
        self.spike_span = int(spike_span)
        self.norm_ratio = float(norm_ratio)
        self.clip_threshold = float(clip_threshold)
        self.clip_fraction_rise = float(clip_fraction_rise)
        self.min_half = max(3, min(int(min_half), self.window // 2))
        self.hold_steps = int(hold_steps)

        self.history = deque(maxlen=max(self.window, self.trend_window))
        self.norm_history = deque(maxlen=self.window)
        self.current_phase = TrainingPhase.WARMUP
        self.phase_start_step = 0
        self.steps_seen = 0
        self.nonfinite_events = 0
        self._hold_until = None
        self._diagnostics = {}

    # ------------------------------------------------------------------
    def step(self, loss: float, step: int, grad_norm: float | None = None) -> TrainingPhase:
        self.steps_seen += 1
        reasons = []
        if _finite(loss):
            self.history.append(float(loss))
        else:
            self.nonfinite_events += 1
            reasons.append("non_finite_loss")
        if grad_norm is None:
            self.norm_history.append(math.nan)
        elif _finite(grad_norm):
            self.norm_history.append(float(grad_norm))
        else:
            self.norm_history.append(math.nan)
            self.nonfinite_events += 1
            reasons.append("non_finite_grad_norm")

        stats = self._statistics()
        if self.steps_seen > self.warmup:
            reasons.extend(self._instability_reasons(stats))
        new_phase = self._decide(step, stats, reasons)

        stats.update(
            step=step,
            steps_seen=self.steps_seen,
            phase=new_phase.value,
            reasons=reasons,
            nonfinite_events=self.nonfinite_events,
            hold_until=self._hold_until,
        )
        self._diagnostics = stats

        if new_phase != self.current_phase:
            duration = step - self.phase_start_step
            self.logger.info(
                f"Training phase: {self.current_phase.value} -> {new_phase.value} "
                f"at step {step} (previous phase lasted {duration} steps; {self._summary(stats)})"
            )
            self.current_phase = new_phase
            self.phase_start_step = step

        return self.current_phase

    def diagnostics(self) -> dict:
        """Statistics behind the latest ``step`` decision (``reasons`` lists fired rules)."""
        return dict(self._diagnostics)

    # ------------------------------------------------------------------
    def _decide(self, step: int, stats: dict, reasons: list) -> TrainingPhase:
        if reasons:
            candidate = TrainingPhase.UNSTABLE
        elif self.steps_seen <= self.warmup:
            candidate = TrainingPhase.WARMUP
        elif self._diverging(stats):
            reasons.append("sustained_rise")
            candidate = TrainingPhase.DIVERGING
        else:
            candidate = self._trend_phase(stats)

        severity_new = _SEVERITY.get(candidate, 0)
        severity_cur = _SEVERITY.get(self.current_phase, 0)
        if severity_new > 0 and severity_new >= severity_cur:
            self._hold_until = step + self.hold_steps
            return candidate
        if severity_cur > 0 and self._hold_until is not None and step < self._hold_until:
            reasons.append("hold")
            return self.current_phase
        return candidate

    def _instability_reasons(self, stats: dict) -> list:
        reasons = []
        disp = stats.get("disp_ratio")
        if disp is not None and disp >= self.dispersion_ratio:
            reasons.append("dispersion_ratio")
        spikes = stats.get("spike_count")
        if spikes is not None and spikes >= self.spike_count:
            reasons.append("spike_cluster")
        ratio = stats.get("norm_ratio")
        if ratio is not None and ratio >= self.norm_ratio:
            reasons.append("grad_norm_ratio")
        rise = stats.get("clip_frac_rise")
        if rise is not None and rise >= self.clip_fraction_rise:
            reasons.append("clip_fraction_rise")
        return reasons

    def _diverging(self, stats: dict) -> bool:
        z = stats.get("z_shift")
        rel = stats.get("rel_change")
        return z is not None and z >= self.diverge_z and rel >= self.diverge_threshold

    def _trend_phase(self, stats: dict) -> TrainingPhase:
        z = stats.get("z_shift")
        if z is None:
            return TrainingPhase.WARMUP
        improvement_z = -z
        improvement = -stats["rel_change"]
        if self.current_phase == TrainingPhase.CONVERGING:
            if improvement_z >= self.converge_exit_z and improvement > 0:
                return TrainingPhase.CONVERGING
        elif improvement_z >= self.converge_z and improvement >= self.plateau_threshold:
            return TrainingPhase.CONVERGING
        return TrainingPhase.PLATEAU

    # ------------------------------------------------------------------
    def _statistics(self) -> dict:
        stats = dict.fromkeys(_STAT_KEYS)
        losses = np.fromiter(self.history, dtype=np.float64, count=len(self.history))

        window = losses[-self.window :]
        stats["n_window"] = int(window.size)
        if window.size >= 2 * self.min_half:
            half = window.size // 2
            older, recent = window[:half], window[half:]
            sigma_older = _winsorized_sigma(_detrend(older))
            sigma_recent = _winsorized_sigma(_detrend(recent))
            stats["sigma_older"] = sigma_older
            stats["sigma_recent"] = sigma_recent
            stats["disp_ratio"] = _ratio(sigma_recent, sigma_older)
            robust = _robust_sigma(_detrend(window))
            stats["sigma_robust"] = robust
            if robust > 0:
                level = float(older.mean()) + self.spike_sigmas * robust
                stats["spike_level"] = level
                stats["spike_count"] = int(np.count_nonzero(window[-self.spike_span :] > level))

        norms = np.fromiter(self.norm_history, dtype=np.float64, count=len(self.norm_history))
        norms = norms[np.isfinite(norms)]
        stats["n_norm"] = int(norms.size)
        if norms.size >= 2 * self.min_half:
            half = norms.size // 2
            older_n, recent_n = norms[:half], norms[half:]
            median_older = float(np.median(older_n))
            median_recent = float(np.median(recent_n))
            stats["norm_median_older"] = median_older
            stats["norm_median_recent"] = median_recent
            stats["norm_ratio"] = _ratio(median_recent, median_older)
            clip_older = float(np.mean(older_n >= self.clip_threshold))
            clip_recent = float(np.mean(recent_n >= self.clip_threshold))
            stats["clip_frac_older"] = clip_older
            stats["clip_frac_recent"] = clip_recent
            stats["clip_frac_rise"] = clip_recent - clip_older

        trend = losses[-self.trend_window :]
        stats["n_trend"] = int(trend.size)
        if trend.size >= 2 * self.min_half:
            half = trend.size // 2
            older_t, recent_t = trend[:half], trend[half:]
            res_older, res_recent = _detrend(older_t), _detrend(recent_t)
            dof = max(older_t.size + recent_t.size - 4, 1)
            pooled = math.sqrt(
                (float(res_older @ res_older) + float(res_recent @ res_recent)) / dof
            )
            se = pooled * math.sqrt(1.0 / older_t.size + 1.0 / recent_t.size)
            older_mean = float(older_t.mean())
            recent_mean = float(recent_t.mean())
            diff = recent_mean - older_mean
            if se > 0:
                z = diff / se
            else:
                z = 0.0 if diff == 0 else math.copysign(math.inf, diff)
            stats.update(
                older_mean=older_mean,
                recent_mean=recent_mean,
                sigma_pooled=pooled,
                se_diff=se,
                rel_change=diff / max(abs(older_mean), 1e-8),
                z_shift=z,
            )
        return stats

    @staticmethod
    def _summary(stats: dict) -> str:
        parts = []
        if stats.get("reasons"):
            parts.append("reasons=" + ",".join(stats["reasons"]))
        formats = (
            ("z_shift", "{:+.2f}"),
            ("rel_change", "{:+.3f}"),
            ("disp_ratio", "{:.2f}"),
            ("spike_count", "{}"),
            ("norm_ratio", "{:.2f}"),
            ("clip_frac_rise", "{:+.2f}"),
        )
        for key, fmt in formats:
            value = stats.get(key)
            if value is not None:
                parts.append(f"{key}={fmt.format(value)}")
        return " ".join(parts) or "no statistics yet"


class _QuietLogger:
    def info(self, *args, **kwargs):
        pass

    warning = error = debug = info


def replay(losses, norms=None, logger=None, **detector_kwargs):
    """Feed a whole loss (and optional norm) series; return (phases, per-step diagnostics)."""
    detector = PhaseDetector(logger or _QuietLogger(), **detector_kwargs)
    phases, diagnostics = [], []
    for index, loss in enumerate(losses):
        norm = None if norms is None else norms[index]
        phases.append(detector.step(loss, index, grad_norm=norm))
        diagnostics.append(detector.diagnostics())
    return phases, diagnostics
