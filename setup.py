"""
To install via pip install -e . in the root of the repository. This will make the BASE_MODELS
package available in the current environment.
"""

from setuptools import find_packages, setup

setup(
    name="base_models",
    packages=find_packages(),
    version="0.0.1",
    description="Base models library needed for a few scientific computing libraries.",
    author="Maxime Rousselet",
)
