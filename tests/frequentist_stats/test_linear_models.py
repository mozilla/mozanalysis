import mozanalysis.frequentist_stats.linear_models as mafslm
import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import CompareMeans


def _make_test_model():
    ref_branch = "treatment-a"
    alphas = [0.01, 0.05]
    branches = ["control"] * 100 + ["treatment-a"] * 100 + ["treatment-b"] * 100
    searches = list(range(100)) + list(range(100, 200)) + list(range(200, 300))
    model_df = pd.DataFrame({"search_count": searches, "branch": branches})
    formula = mafslm._make_formula("search_count", ref_branch)
    results = smf.ols(formula, model_df).fit()

    return results, alphas, ref_branch, model_df


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

    mean = 49.5
    low, high = 43.74349, 55.25650

    assert np.isclose(actuals["exp"], mean)
    assert np.isclose(actuals["0.5"], mean)
    assert np.isclose(actuals["0.025"], low)
    assert np.isclose(actuals["0.975"], high)

    index_values = actuals.index.values
    index_values.sort()
    assert list(index_values) == ["0.025", "0.5", "0.975", "exp"]


def test_summarize_univariate():
    one_branch_data = pd.Series(range(100))
    test_data = pd.concat([one_branch_data, one_branch_data])
    branches = pd.Series([*(["control"] * 100), *(["treatment"] * 100)])
    branch_list = ["control", "treatment"]

    result = mafslm.summarize_univariate(test_data, branches, branch_list, [0.05])

    assert sorted(result.keys()) == branch_list

    mean = 49.5
    low, high = 43.74349, 55.25650

    for branch in branch_list:
        assert np.isclose(result[branch]["exp"], mean)
        assert np.isclose(result[branch]["0.5"], mean)
        assert np.isclose(result[branch]["0.025"], low)
        assert np.isclose(result[branch]["0.975"], high)

        index_values = result[branch].index.values
        index_values.sort()
        assert list(index_values) == ["0.025", "0.5", "0.975", "exp"]


def test__make_formula():
    expected = "search_count ~ C(branch, Treatment(reference='control'))"
    actual = mafslm._make_formula("search_count", "control")

    assert expected == actual

    expected = "days_of_use ~ C(branch, Treatment(reference='treatment-a'))"
    actual = mafslm._make_formula("days_of_use", "treatment-a")

    assert expected == actual

    expected = (
        "active_hours ~ C(branch, Treatment(reference='treatment-a'))"
        " + active_hours_pre"
    )
    actual = mafslm._make_formula("active_hours", "treatment-a", "active_hours_pre")

    assert expected == actual


def test__make_joint_output():
    out = mafslm._make_joint_output([0.01, 0.05], "rel_uplift")

    expected_keys = [
        ("rel_uplift", "exp"),
        ("rel_uplift", "0.5"),
        ("rel_uplift", "0.005"),
        ("rel_uplift", "0.995"),
        ("rel_uplift", "0.025"),
        ("rel_uplift", "0.975"),
    ]

    for key in expected_keys:
        assert key in out.index

    assert len(out.index) == len(expected_keys)


def test__extract_absolute_uplifts():
    results, alphas, ref_branch, model_df = _make_test_model()

    # control vs treatment-a
    out = mafslm._extract_absolute_uplifts(results, "control", ref_branch, alphas)

    assert np.isclose(out[("abs_uplift", "exp")], -100)
    assert np.isclose(out[("abs_uplift", "0.5")], -100)

    for alpha in alphas:
        a_str_low, a_str_high = mafslm.stringify_alpha(alpha)
        low, high = CompareMeans.from_data(
            model_df.loc[model_df.branch == "control", "search_count"],
            model_df.loc[model_df.branch == ref_branch, "search_count"],
        ).zconfint_diff(alpha)
        assert np.isclose(out[("abs_uplift", a_str_low)], low, atol=0.1)
        assert np.isclose(out[("abs_uplift", a_str_high)], high, atol=0.1)

    # treatment-a vs treatment-b
    out = mafslm._extract_absolute_uplifts(results, "treatment-b", ref_branch, alphas)

    assert np.isclose(out[("abs_uplift", "exp")], 100)
    assert np.isclose(out[("abs_uplift", "0.5")], 100)

    for alpha in alphas:
        a_str_low, a_str_high = mafslm.stringify_alpha(alpha)
        low, high = CompareMeans.from_data(
            model_df.loc[model_df.branch == "treatment-b", "search_count"],
            model_df.loc[model_df.branch == ref_branch, "search_count"],
        ).zconfint_diff(alpha)
        assert np.isclose(out[("abs_uplift", a_str_low)], low, atol=0.1)
        assert np.isclose(out[("abs_uplift", a_str_high)], high, atol=0.1)
