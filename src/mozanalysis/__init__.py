# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

# This allows wildcard imports: from mozanalysis import *
# Generally, import * should be avoided, however it is used by Jetstream
# We use it to expose modules available in mozanalysis package.
# https://github.com/mozilla/jetstream
import os
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mozanalysis")
except PackageNotFoundError:
    __version__ = "unknown version"

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

# lookup app_name via app_id
# used for backwards compatibility
APPS = {
    "firefox_desktop": {"firefox_desktop"},
    "fenix": {"org_mozilla_fenix"},
    "focus_android": {"org_mozilla_focus"},
    "firefox_ios": {"org_mozilla_ios_FirefoxBeta"},
    "focus_ios": {"org_mozilla_ios_Focus"},
    "klar_ios": {"org_mozilla_ios_Klar"},
}
