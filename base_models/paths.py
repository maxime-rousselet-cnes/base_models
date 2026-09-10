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


MODELS = {
    "elastic": "PREM",
    "attenuation": "Resovsky_upper_uniform_lower",
    "transient": "Post_sismo_upper_uniform_lower",
    "viscous": "VM7",
}
SOLID_EARTH_MODEL_PROFILES = MODELS.keys()

# Contains both inputs and outputs.
DATA_PATH_TXT_PATH = Path("..")

with open(DATA_PATH_TXT_PATH.joinpath("data_path.txt"), "r", encoding="utf-8") as file:

    ROOT_PATH = Path("".join((line.strip() for line in file.readlines())))

DATA_PATH = ROOT_PATH.joinpath("common_data")
TEST_PATH = DATA_PATH.joinpath("tests")
FIGURES_PATH = DATA_PATH.joinpath("figures")
