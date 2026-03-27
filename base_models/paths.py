"""
Arborescence and common constants configuration.
"""

from enum import Enum
from pathlib import Path


class SolidEarthModelPart(Enum):
    """
    Available model parts.
    """

    attenuation = "attenuation"
    elastic = "elastic"
    transient = "transient"
    viscous = "viscous"


DEFAULT_MODELS: dict[str, str] = {
    "elastic": "PREM",
    "attenuation": "Resovsky",
    "transient": "reference",
    "viscous": "VM7",
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

## Outputs.
OUTPUTS_PATH = DATA_PATH.joinpath("outputs")

### Love numbers.
LOVE_NUMBERS_PATH = OUTPUTS_PATH.joinpath("love_numbers")

### Parallel computing logs.
LOGS_PATH = OUTPUTS_PATH.joinpath("logs")
