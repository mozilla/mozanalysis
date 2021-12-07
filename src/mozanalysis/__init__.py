# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# This allows wildcard imports: from mozanalysis.metrics import *
# Generally, import * should be avoided, however it is needed by Jetstream
# https://github.com/mozilla/jetstream
import os

__all__ = []
basedir = os.path.dirname(__file__)
for root, subdirs, files in os.walk(basedir):
    __all__ += [module for module in subdirs if not module.startswith("_")]
    module_files = [
        file.replace(".py", "")
        for file in files
        if file.endswith(".py") and not file.startswith("_")
    ]
    if os.path.basename(root) in __all__:
        module_files = [os.path.basename(root) + "." + f for f in module_files]
    __all__ += module_files
