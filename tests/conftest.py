# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import logging
import os
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), "helpers"))


def pytest_configure():

    logger = logging.getLogger("py4j")
    logger.setLevel(logging.ERROR)
