"""
Arborescenceand common constants configuration.
"""

from pathlib import Path

DEFAULT_MODELS: dict[str, str] = {
    "ELASTIC": "PREM",
    "ATTENUATION": "Resovsky",
    "TRANSIENT": "reference",
    "VISCOUS": "VM7",
}
SOLID_EARTH_MODEL_PROFILES = DEFAULT_MODELS.keys()

# Contains both inputs and outputs.
DATA_PATH = Path("../COMMON_DATA")

## Tests.
TEST_PATH = DATA_PATH.joinpath("TESTS")

## Inputs.
INPUTS_PATH = DATA_PATH.joinpath("INPUTS")

### Solid Earth model descriptions.
SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_ROOT_PATH = INPUTS_PATH.joinpath(
    "SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS"
)
SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_PATH: dict[str, Path] = {
    model_part: SOLID_EARTH_MODEL_PROFILE_DESCRIPTIONS_ROOT_PATH.joinpath(model_part)
    for model_part in SOLID_EARTH_MODEL_PROFILES
}

## Solid Earth numerical models.
SOLID_EARTH_NUMERICAL_MODELS_ROOT_PATH = DATA_PATH.joinpath("SOLID_EARTH_NUMERICAL_MODELS")
SOLID_EARTH_NUMERICAL_MODELS_PATH: dict[str, Path] = {
    model_part: SOLID_EARTH_NUMERICAL_MODELS_ROOT_PATH.joinpath("COMPONENTS").joinpath(model_part)
    for model_part in SOLID_EARTH_MODEL_PROFILES
}
SOLID_EARTH_FULL_NUMERICAL_MODELS_PATH = SOLID_EARTH_NUMERICAL_MODELS_ROOT_PATH.joinpath(
    "FULL_NUMERICAL_MODELS"
)

## Outputs.
OUTPUTS_PATH = DATA_PATH.joinpath("OUTPUTS")

### Love numbers.
LOVE_NUMBERS_PATH = OUTPUTS_PATH.joinpath("LOVE_NUMBERS")

### Parallel computing logs.
LOGS_PATH = OUTPUTS_PATH.joinpath("LOGS")


def get_love_numbers_subpath(model_id: str, n: int, period: float) -> Path:
    """
    Generates the path to save the y_i system integration results for a given model.
    """
    return (
        LOVE_NUMBERS_PATH.joinpath("INDIVIDUAL_LOVE_NUMBERS")
        .joinpath(model_id)
        .joinpath(str(n))
        .joinpath(str(period))
    )


def get_interpolated_love_numbers_subpath(periods_id: str, rheological_model_id: str) -> Path:
    """
    Gets the path for Love numbers of a given rheological model interpolated on given periods.
    """

    return (
        LOVE_NUMBERS_PATH.joinpath("INTERPOLATED_LOVE_NUMBERS")
        .joinpath(periods_id)
        .joinpath(rheological_model_id)
    )
