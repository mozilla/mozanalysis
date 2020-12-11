# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


def enumerate_included(modules, klass):
    collected = []
    for module in modules:
        collected.extend(
            [(k, v) for k, v in module.__dict__.items() if isinstance(v, klass)]
        )
    return collected
