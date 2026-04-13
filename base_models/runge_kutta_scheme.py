"""
Custom RK45 numerical integration scheme thread-safe and multiprocess-safe.
"""

from typing import Callable, Optional

from numpy import array, inf, isinf, isnan, maximum, ndarray

FIRST_STEP_FACTOR = 1e-10

# Dormand-Prince coefficients for RK45.
DOPRI_C = array(object=[0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0], dtype=float)
DOPRI_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
DOPRI_B = array(object=[35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=float)
DOPRI_B_ALT = array(
    object=[5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40],
    dtype=float,
)


def compute_stages(
    fun: Callable,
    t: float,
    step: float,
    y: ndarray,
    arguments: tuple[Optional[tuple[ndarray, ndarray]], tuple] = (None, ()),
) -> list[ndarray]:
    """
    Compute intermediate RK45 stages using Dormand-Prince coefficients.
    Optinoally handles a parameterized function.
    """

    stages = []
    parameters, arguments = arguments

    for i in range(7):

        t_i = t + DOPRI_C[i] * step
        y_i = y.copy()

        for j, stage in enumerate(stages[:i]):

            y_i += step * DOPRI_A[i][j] * stage

        if parameters is None:

            stages.append(array(object=fun(t_i, y_i, *arguments), dtype=y_i.dtype))

        else:

            p_start, p_end = parameters
            p_i = p_start + DOPRI_C[i] * (p_end - p_start)
            stages.append(array(object=fun(t_i, p_i, y_i, *arguments), dtype=y_i.dtype))

    return stages


def estimate_solution(y: ndarray, stages: list[ndarray], step: float, b_coeffs: ndarray) -> ndarray:
    """
    Estimate solution using given Butcher tableau weights.
    """

    y_estimate = y.copy()

    for i, stage in enumerate(stages):

        y_estimate += step * b_coeffs[i] * stage

    return y_estimate


def compute_error_ratio(
    y: ndarray,
    y_high: ndarray,
    y_low: ndarray,
    rtol: float,
    atol: float,
) -> float:
    """
    Compute RK45 error ratio for adaptive step size control.
    """

    scale = atol + rtol * maximum(abs(y), abs(y_high))

    return max(abs(y_high - y_low) / scale)


def adaptive_runge_kutta_45(
    fun: Callable[[float, ndarray], ndarray],
    t_bounds: tuple[float, float, Optional[float]],  # t_0, t_end, max_dt.
    y_0: ndarray,
    # The solver keeps the local error estimates under atol + rtol * abs(yr).
    tolerances: tuple[float, float] = (1.0e-14, 1.0e-10),  # atol, rtol.
    arguments: tuple = (),
) -> tuple[ndarray, ndarray]:
    """
    Adaptive Runge-Kutta-Fehlberg (RK45) ODE solver with overflow and instability protection.
    """

    _, t_end, max_dt = t_bounds
    t = [t_bounds[0]]
    step = max_dt if max_dt is not None else (t_end - t_bounds[0]) / FIRST_STEP_FACTOR
    max_step = (t_end - t_bounds[0]) / 2
    max_step = max_step if max_dt is None else min(max_dt, max_step)
    y = [y_0]

    while t[-1] < t_bounds[1]:

        if t[-1] + step > t_end:

            step = t_end - t[-1]

        stages = compute_stages(fun=fun, t=t[-1], step=step, y=y[-1], arguments=(None, arguments))
        y_high = estimate_solution(y=y[-1], stages=stages, step=step, b_coeffs=DOPRI_B)
        y_low = estimate_solution(y=y[-1], stages=stages, step=step, b_coeffs=DOPRI_B_ALT)

        if inf in y_high:

            raise OverflowError(f"y_high overflowed at t={t[-1]}, step={step}, y={y[-1]}")

        error_ratio = compute_error_ratio(
            y=y[-1], y_high=y_high, y_low=y_low, rtol=tolerances[1], atol=tolerances[0]
        )

        if isnan(error_ratio) or isinf(error_ratio):

            raise RuntimeError(f"Unstable integration: error_ratio={error_ratio}")

        if error_ratio <= 1.0:

            t.append(t[-1] + step)
            y.append(y_high)

        # Default RK45 step factor adjustment with safety margin.
        step = min(
            step
            * min(
                4.0, max(0.1, 0.9 * error_ratio ** (-0.25) if error_ratio > tolerances[0] else 2.0)
            ),
            max_step,
        )
        step = step if max_dt is None else min(max_dt, step)

    return array(object=t), array(object=y)


def non_adaptive_runge_kutta_45(
    fun: Callable[[float, ndarray, ndarray], ndarray],
    t: ndarray,
    dy_dgamma_0: ndarray,
    y: ndarray,
    arguments: tuple = (),
) -> ndarray:
    """
    Non-adaptive RK45 solver using Dormand-Prince coefficients for a parameterized function.
    Computes solution at the given t points.
    """

    if len(t) < 2:

        return array(object=[dy_dgamma_0])

    dy_dgamma = dy_dgamma_0
    dy_dgamma_tab = [dy_dgamma]

    for k in range(1, len(t)):

        t_prev = t[k - 1]
        step = t[k] - t_prev
        stages = compute_stages(
            fun=fun,
            t=t_prev,
            step=step,
            y=dy_dgamma,
            arguments=((y[k - 1], y[k]), arguments),
        )
        dy_dgamma = estimate_solution(y=dy_dgamma, stages=stages, step=step, b_coeffs=DOPRI_B)
        dy_dgamma_tab.append(dy_dgamma)

    return array(object=dy_dgamma_tab)
