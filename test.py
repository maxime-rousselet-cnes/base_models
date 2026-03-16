"""
Tests the BASE_MODELS package implementation.
To be called by pytest test.py
"""

from numpy import array

from base_models import (
    TEST_PATH,
    load_base_model,
    load_complex_array,
    save_base_model,
    save_complex_array,
)

CONSISTENCY_TOLERANCE = 1e-10


def test_save_and_load_base_model(file_name: str = "save_base_model_test_file") -> None:
    """
    Saves a base model in a (.JSON) file and verifies oconsistency when loading.
    """

    obj = {
        "a": 1,
        "b": "test",
        "c": [1, 2, 3],
    }

    save_base_model(obj=obj, name=file_name, path=TEST_PATH)
    loaded_obj = load_base_model(name=file_name, path=TEST_PATH)

    assert obj == loaded_obj


def test_save_and_load_complex_array(file_name: str = "save_complex_array_test_file") -> None:
    """
    Saves a complex array in a (.JSON) file and verifies oconsistency when loading.
    """

    obj = array([1.0 + 2.0j, 3.0 + 4.0j], dtype=complex)

    save_complex_array(obj=obj, name=file_name, path=TEST_PATH)
    loaded_obj = load_complex_array(name=file_name, path=TEST_PATH)

    assert sum(abs(obj - loaded_obj)) < CONSISTENCY_TOLERANCE
