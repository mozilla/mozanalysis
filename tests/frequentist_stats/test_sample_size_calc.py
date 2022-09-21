# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd

from mozanalysis.frequentist_stats.sample_size import z_or_t_ind_sample_size_calc
from mozanalysis.metrics.desktop import search_clients_daily, uri_count


def test_sample_size_calc_desktop():
    df = pd.DataFrame(
        {
            search_clients_daily.name: np.random.normal(size=100),
            uri_count.name: np.random.normal(size=100),
        }
    )

    res = z_or_t_ind_sample_size_calc(df, [search_clients_daily, uri_count])

    assert all([c in res.keys() for c in df.columns])

    assert res[search_clients_daily.name]["sample_size"] > 1000000
    assert res[uri_count.name]["sample_size"] > 1000000
