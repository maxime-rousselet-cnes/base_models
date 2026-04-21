"""
Usefull functions to manipulate Fourier transforms of steady-state regime signals.
"""

from dataclasses import dataclass

from numpy import arange, concatenate, linspace, ndarray, zeros_like
from scipy.interpolate import lagrange


@dataclass
class SteadyStateSignalParameters:
    cubic_spline_length: float = 50.0
    plateau_length: float = 500.0


def cubic_spline_connection(
    signal_1: ndarray, signal_2: ndarray, cubic_spline_length: int
) -> ndarray:
    """
    Connect two signals with a spline of num_points points.
    """

    t = linspace(start=0, stop=1, num=cubic_spline_length + 1)
    h00 = (
        2 * t**3 - 3 * t**2 + 1
    )  # Hermite polynomial with h00(0)=1, h00(1)=0, h00'(0)=0, h00'(1)=0.
    h01 = -2 * t**3 + 3 * t**2  # Hermite polynomial with h01(0)=0, h01(1)=1, h01'(0)=0, h01'(1)=0.
    bridge = h00 * signal_1[-1] + h01 * signal_2[0]

    return concatenate([signal_1, bridge[1:-1], signal_2])


def make_antisymmetric(signal: ndarray, cubic_spline_length: int) -> ndarray:
    """
    Returns an anti-symmetric signal connected via cubic spline.
    """

    return cubic_spline_connection(
        signal_1=signal, signal_2=-signal[::-1], cubic_spline_length=cubic_spline_length
    )


def pad_signal(signal: ndarray, plateau_length: int) -> ndarray:
    """
    Pad signal with zeros.
    """

    return concatenate(([0] * plateau_length, signal))


def build_steady_state_regime_signal(
    t: ndarray, signal: ndarray, plateau_length: float, cubic_spline_length: float
) -> tuple[int, ndarray, ndarray]:
    """
    Build a steady-state regime model of a given signal.
    """

    time_step = t[1] - t[0]
    plateau_length_in_samples = int(plateau_length / time_step)
    cubic_spline_length_in_samples = int(cubic_spline_length / time_step)

    return (
        plateau_length_in_samples,
        concatenate(
            (
                arange(
                    start=t[0] - plateau_length_in_samples * time_step, stop=t[0], step=time_step
                ),
                t,
                arange(
                    start=t[-1] + time_step,
                    stop=t[-1]
                    + (cubic_spline_length_in_samples + len(t) + plateau_length_in_samples)
                    * time_step,
                    step=time_step,
                ),
            )
        ),
        make_antisymmetric(
            signal=pad_signal(signal=signal, plateau_length=plateau_length_in_samples),
            cubic_spline_length=cubic_spline_length_in_samples,
        ),
    )


def lagrange_order4(x: ndarray, y: ndarray, new_x: ndarray) -> ndarray:
    """
    Order-4 Lagrange interpolation (5-point) with automatic selection.
    """

    n = len(x)
    new_y = zeros_like(a=new_x, dtype=float)
    i = 0

    for j, nx in enumerate(new_x):

        while i + 1 < n and x[i + 1] <= nx:

            i += 1

        idx_start = i - 2
        idx_end = idx_start + 5

        if idx_start < 0:

            idx_start = 0
            idx_end = 5

        if idx_end > n:

            idx_end = n
            idx_start = n - 5

        idx_start = max(idx_start, 0)
        x_window = x[idx_start:idx_end]
        y_window = y[idx_start:idx_end]
        poly = lagrange(x_window, y_window)
        new_y[j] = poly(nx)

    return new_y
