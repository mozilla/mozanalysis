#!/usr/bin/env python
# encoding: utf-8

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from setuptools import setup, find_packages

tests_require = [
    'mock', 'pytest', 'pytest-cov', 'pytest-timeout', 'pyspark'
]

docs_require = ['Sphinx', 'sphinx-autobuild', 'sphinx-rtd-theme']

setup(
    name='mozanalysis',
    use_scm_version=True,
    author='Rob Hudson',
    author_email='robhudson@mozilla.com',
    description='A library for Mozilla experiments analysis',
    url='https://github.com/mozilla/mozanalysis',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'attrs',
        'numpy',
        'pandas',
        'scipy',
        'google-cloud-bigquery',
    ],
    setup_requires=['pytest-runner', 'setuptools_scm'],
    extras_require={
        'testing': tests_require,
        'docs': docs_require,
    },
    tests_require=tests_require,
)
