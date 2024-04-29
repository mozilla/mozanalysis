#!/usr/bin/env python

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from setuptools import find_packages, setup

tests_require = [
    "mock",
    "pytest",
    "pytest-ruff",
    "pytest-cov",
    "pytest-timeout",
    "ruff==0.3.1",
]

docs_require = ["Sphinx", "sphinx-autobuild", "sphinx-rtd-theme"]

viz_require = [
    "pandas-gbq",
    "mozilla-nimbus-schemas",
    "nbformat",
    "nbconvert",
    "requests",
    "Click",
    "ipykernel",
]

setup(
    name="mozanalysis",
    use_scm_version=True,
    author="Mozilla Corporation",
    author_email="fx-data-dev@mozilla.org",
    description="A library for Mozilla experiments analysis",
    url="https://github.com/mozilla/mozanalysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10,<=3.11",
    install_requires=[
        "attrs",
        "mozilla-metric-config-parser",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "google-cloud-bigquery",
        "google-cloud-bigquery-storage",
        "statsmodels",
        "matplotlib",
    ],
    setup_requires=["pytest-runner", "setuptools_scm"],
    extras_require={
        "testing": tests_require,
        "docs": docs_require,
        "vizualization": viz_require,
    },
    tests_require=tests_require,
    entry_points={"console_scripts": ["mozanalysis = mozanalysis.cli:cli"]},
)
