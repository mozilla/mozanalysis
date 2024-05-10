import mozanalysis.frequentist_stats.linear_models as mafslm
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_1samp


def test_stringify_alpha():
    for bad_alphas in [-1, 0, 1, 2]:
        with pytest.raises(ValueError, match=r"alpha must be in \(0,1\)"):
            mafslm.stringify_alpha(bad_alphas)

    alpha = 0.1
    low, high = mafslm.stringify_alpha(alpha)
    assert low == "0.050"
    assert high == "0.950"

    alpha = 0.05
    low, high = mafslm.stringify_alpha(alpha)
    assert low == "0.025"
    assert high == "0.975"

    alpha = 0.01
    low, high = mafslm.stringify_alpha(alpha)
    assert low == "0.005"
    assert high == "0.995"

def test_summarize_one_branch():
    test_data = pd.Series(range(100))
    alphas = [0.05]
    actuals = mafslm.summarize_one_branch(test_data, alphas)

    mean = np.mean(test_data)

    assert np.isclose(actuals["exp"], mean)
    assert np.isclose(actuals["0.5"], mean)

    ttest_result = ttest_1samp(test_data, mean)
    low, high = ttest_result.confidence_interval(1-alphas[0])
    assert np.isclose(actuals["0.025"], low)
    assert np.isclose(actuals["0.975"], high)

    index_values = actuals.index.values
    index_values.sort()
    assert list(index_values) == ["0.025", "0.5", "0.975", "exp", "mean"]
