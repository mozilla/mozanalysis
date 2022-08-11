# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import pytest

from mozanalysis.frequentist_stats import sample_size_calc
from mozanalysis.metrics.desktop import search_clients_daily, uri_count

def test_sample_size_calc_desktop():
    df = pd.DataFrame({
        search_clients_daily.name: np.random.normal(size=100),
        uri_count.name: np.random.normal(size=100)
    })

    res = sample_size_calc(df, [search_clients_daily, uri_count])

    assert(all([df.columns in res.keys()]))

    assert(res[search_clients_daily.name] > 1000000)
    assert(res[uri_count.name] > 1000000)



