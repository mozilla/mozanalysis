# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import logging

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.master("local").appName("mozanalysis_test").getOrCreate()
    )

    logger = logging.getLogger("py4j")
    logger.setLevel(logging.ERROR)

    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def spark_context(spark):
    return spark.sparkContext
