#!/usr/bin/env python
# encoding: utf-8

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from setuptools import setup, find_packages

tests_require = [
    "mock",
    "pytest",
    "pytest-black",
    "pytest-cov",
    "pytest-timeout",
    "pyspark",
]

docs_require = ["Sphinx", "sphinx-autobuild", "sphinx-rtd-theme"]

setup(
    name="mozanalysis",
    use_scm_version=True,
    author="Mozilla Corporation",
    author_email="fx-data-dev@mozilla.org",
    description="A library for Mozilla experiments analysis",
    url="https://github.com/mozilla/mozanalysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "attrs",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "google-cloud-bigquery",
        "google-cloud-bigquery-storage",
    ],
    setup_requires=["pytest-runner", "setuptools_scm"],
    extras_require={
        "testing": tests_require,
        "docs": docs_require,
    },
    tests_require=tests_require,
)
