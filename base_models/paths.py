"""
Arborescence and common constants configuration.
"""

from enum import Enum
from pathlib import Path


class SolidEarthModelPart(Enum):
    """
    Available model parts.
    """

    ATTENUATION = "attenuation"
    ELASTIC = "elastic"
    TRANSIENT = "transient"
    VISCOUS = "viscous"


DEFAULT_MODELS: dict[str, str] = {
    "elastic": "PREM",
    "attenuation": "uniform",
    "transient": "reference",
    "viscous": "uniform",
}
SOLID_EARTH_MODEL_PROFILES = DEFAULT_MODELS.keys()

# Contains both inputs and outputs.
DATA_PATH = Path("../common_data")

## Tests.
TEST_PATH = DATA_PATH.joinpath("tests")

### Test figures.
TEST_FIGURES_PATH = TEST_PATH.joinpath("figures")

## Inputs.
INPUTS_PATH = DATA_PATH.joinpath("inputs")

## Love numbers.
LOVE_NUMBERS_PATH = DATA_PATH.joinpath("love_numbers")
