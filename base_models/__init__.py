"""
Base models library needed for a few scientific computing libraries.
"""

from enum import Enum

from .database import load_base_model, load_complex_array, save_base_model, save_complex_array
from .paths import TEST_PATH


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

to_import = [load_base_model, load_complex_array, save_base_model, save_complex_array, TEST_PATH]
