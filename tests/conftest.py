# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import logging

import sys
import os
import pytest
from pyspark.sql import SparkSession


sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))

_spark_session = None
_spark_context = None


def pytest_configure(config):
    global _spark_session, _spark_context

    _spark_session = (
        SparkSession.builder.master("local").appName("mozanalysis_test").getOrCreate()
    )
    _spark_context = _spark_session.sparkContext

    logger = logging.getLogger("py4j")
    logger.setLevel(logging.ERROR)


def pytest_unconfigure(config):
    _spark_session.stop()


@pytest.fixture()
def spark():
    return _spark_session


@pytest.fixture()
def spark_context(spark):
    return _spark_context


def pytest_generate_tests(metafunc):
    if 'spark_context_or_none' in metafunc.fixturenames:
        metafunc.parametrize(
            "spark_context_or_none", ["spark", "no spark"], indirect=True
        )


@pytest.fixture()
def spark_context_or_none(request):
    if request.param == 'spark':
        return _spark_context
    elif request.param == 'no spark':
        return None
    else:
        raise ValueError("invalid internal test config")
