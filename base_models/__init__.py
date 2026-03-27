"""
Base models library needed for a few scientific computing libraries.
"""

from enum import Enum

from .database import load_base_model, load_complex_array, save_base_model, save_complex_array
from .paths import (
    DATA_PATH,
    DEFAULT_MODELS,
    LOGS_PATH,
    LOVE_NUMBERS_PATH,
    SOLID_EARTH_MODEL_PROFILES,
    TEST_FIGURES_PATH,
    TEST_PATH,
    SolidEarthModelPart,
)
from .runge_kutta_scheme import adaptive_runge_kutta_45, non_adaptive_runge_kutta_45
from .symbolic import (
    evaluate_terminal_parameters,
    fixed_timestep_integrator,
    partial_symbols,
    variation_equation,
    vector_variation_equation,
)


class Direction(Enum):
    """
    Love numbers directions.
    """

    VERTICAL = 0
    TANGENTIAL = 1
    POTENTIAL = 2


class BoundaryCondition(Enum):
    """
    Love numbers boundary conditions.
    """

    LOAD = 0
    SHEAR = 1
    POTENTIAL = 2


# Earth mean radius (m).
EARTH_RADIUS = 6.371e6

to_import = [
    load_base_model,
    load_complex_array,
    save_base_model,
    save_complex_array,
    DATA_PATH,
    DEFAULT_MODELS,
    LOGS_PATH,
    LOVE_NUMBERS_PATH,
    SOLID_EARTH_MODEL_PROFILES,
    TEST_FIGURES_PATH,
    TEST_PATH,
    SolidEarthModelPart,
    adaptive_runge_kutta_45,
    non_adaptive_runge_kutta_45,
    evaluate_terminal_parameters,
    fixed_timestep_integrator,
    partial_symbols,
    variation_equation,
    vector_variation_equation,
]
