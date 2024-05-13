import mozanalysis.frequentist_stats.bootstrap as mafsb
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

    return results, alphas, ref_branch, model_df, "search_count"

def _make_test_model_covariate():
    ref_branch = "treatment-a"
    alphas = [0.01, 0.05]
    branches = ["control"] * 100 + ["treatment-a"] * 100 + ["treatment-b"] * 100
    y_base = np.random.normal(loc=2, scale=0.1, size = 300)
    y_pre_adj = np.random.normal(loc=0, scale=0.05, size = 300)
    te = np.concatenate([
        np.random.normal(loc=0, scale = 0.1, size = 100),
        np.random.normal(loc=0.1, scale = 0.1, size = 100),
        np.random.normal(loc=0.2, scale = 0.1, size = 100),
    ])
    model_df = pd.DataFrame({
        "search_count":y_base+te,
        "branch":branches,
        "search_count_pre":y_base+y_pre_adj
    })
    formula = mafslm._make_formula("search_count", ref_branch, "search_count_pre")
    results = smf.ols(formula, model_df).fit()

    return results, alphas, ref_branch, model_df, "search_count", "search_count_pre"


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

    assert np.isclose(actuals["0.5"], mean)
    assert np.isclose(actuals["mean"], mean)
    assert np.isclose(actuals["0.025"], low)
    assert np.isclose(actuals["0.975"], high)

    index_values = actuals.index.values
    index_values.sort()
    assert list(index_values) == ["0.025", "0.5", "0.975", "mean"]


def test_summarize_univariate():
    one_branch_data = pd.Series(range(100))
    test_data = pd.concat([one_branch_data, one_branch_data])
    branches = pd.Series([*(["control"] * 100), *(["treatment"] * 100)])
    branch_list = ["control", "treatment"]

    result = mafslm.summarize_univariate(test_data, branches, branch_list, [0.05])

    assert sorted(result.keys()) == branch_list

    # validate against theoretical values
    mean = 49.5
    low, high = 43.74349, 55.25650
    for branch in branch_list:
        assert np.isclose(result[branch]["mean"], mean)
        assert np.isclose(result[branch]["0.5"], mean)
        assert np.isclose(result[branch]["0.025"], low)
        assert np.isclose(result[branch]["0.975"], high)

        index_values = result[branch].index.values
        index_values.sort()
        assert list(index_values) == ["0.025", "0.5", "0.975", "mean"]

    # cross-validate against existing bootstrap implementation
    bootstrap_result = mafsb.bootstrap_one_branch(one_branch_data)
    for branch in branch_list:
        assert np.isclose(result[branch]["mean"], bootstrap_result["mean"], atol=0.5)
        assert np.isclose(result[branch]["0.5"], bootstrap_result["0.5"], atol=0.5)
        assert np.isclose(result[branch]["0.025"], bootstrap_result["0.025"], atol=0.5)
        assert np.isclose(result[branch]["0.975"], bootstrap_result["0.975"], atol=0.5)


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
    results, alphas, ref_branch, model_df, column_label = _make_test_model()

    bootstrap_results = mafsb.compare_branches(
        model_df, column_label, ref_branch_label=ref_branch
    )

    # control vs treatment-a
    out = mafslm._extract_absolute_uplifts(results, "control", ref_branch, alphas)

    expected_mean = 50 - 150
    assert np.isclose(out[("abs_uplift", "exp")], expected_mean)
    assert np.isclose(out[("abs_uplift", "0.5")], expected_mean)

    for alpha in alphas:
        a_str_low, a_str_high = mafslm.stringify_alpha(alpha)
        low, high = CompareMeans.from_data(
            model_df.loc[model_df.branch == "control", column_label],
            model_df.loc[model_df.branch == ref_branch, column_label],
        ).zconfint_diff(alpha)
        assert np.isclose(out[("abs_uplift", a_str_low)], low, atol=0.1)
        assert np.isclose(out[("abs_uplift", a_str_high)], high, atol=0.1)

    ## cross-validate with existing bootstrap implementation
    boot_df = bootstrap_results["comparative"]["control"]
    for alpha in alphas:
        a_str_low, a_str_high = mafslm.stringify_alpha(alpha)
        assert np.isclose(
            out[("abs_uplift", a_str_low)], boot_df[("abs_uplift", a_str_low)], atol=1.0
        )
        assert np.isclose(
            out[("abs_uplift", a_str_high)],
            boot_df[("abs_uplift", a_str_high)],
            atol=1.0,
        )
    # treatment-a vs treatment-b
    out = mafslm._extract_absolute_uplifts(results, "treatment-b", ref_branch, alphas)

    expected_mean = 250 - 150
    assert np.isclose(out[("abs_uplift", "exp")], expected_mean)
    assert np.isclose(out[("abs_uplift", "0.5")], expected_mean)

    for alpha in alphas:
        a_str_low, a_str_high = mafslm.stringify_alpha(alpha)
        low, high = CompareMeans.from_data(
            model_df.loc[model_df.branch == "treatment-b", column_label],
            model_df.loc[model_df.branch == ref_branch, column_label],
        ).zconfint_diff(alpha)
        assert np.isclose(out[("abs_uplift", a_str_low)], low, atol=0.1)
        assert np.isclose(out[("abs_uplift", a_str_high)], high, atol=0.1)

    ## cross-validate with existing bootstrap implementation
    boot_df = bootstrap_results["comparative"]["treatment-b"]
    for alpha in alphas:
        a_str_low, a_str_high = mafslm.stringify_alpha(alpha)
        assert np.isclose(
            out[("abs_uplift", a_str_low)], boot_df[("abs_uplift", a_str_low)], atol=1.0
        )
        assert np.isclose(
            out[("abs_uplift", a_str_high)],
            boot_df[("abs_uplift", a_str_high)],
            atol=1.0,
        )

def test__extract_absolute_uplifts_covariate():
    results, alphas, ref_branch, model_df, column_label, column_label_pre = _make_test_model_covariate()

    bootstrap_results = mafsb.compare_branches(
        model_df, column_label, ref_branch_label=ref_branch
    )

    # control vs treatment-a
    out = mafslm._extract_absolute_uplifts(results, "control", ref_branch, alphas)

    expected_mean = -0.1
    assert np.isclose(out[("abs_uplift", "exp")], expected_mean, atol=0.05)
    assert np.isclose(out[("abs_uplift", "0.5")], expected_mean, atol=0.05)

    ## compare to bootstrap implementation
    # validate the point estimates are more precise
    boot_df = bootstrap_results["comparative"]["control"]

    ###validate the widths of confidence intervals are smaller with LM approach
    for alpha in alphas:
        a_str_low, a_str_high = mafslm.stringify_alpha(alpha)
        ci_width_boot = boot_df[("abs_uplift", a_str_high)] - boot_df[("abs_uplift", a_str_low)]
        ci_width_lm = out[("abs_uplift", a_str_high)] - out[("abs_uplift", a_str_low)]
        assert ci_width_lm < ci_width_boot

    # treatment-a vs treatment-b
    out = mafslm._extract_absolute_uplifts(results, "treatment-b", ref_branch, alphas)

    expected_mean = 0.1
    assert np.isclose(out[("abs_uplift", "exp")], expected_mean, atol=0.05)
    assert np.isclose(out[("abs_uplift", "0.5")], expected_mean, atol=0.05)

    boot_df = bootstrap_results["comparative"]["treatment-b"]
    for alpha in alphas:
        a_str_low, a_str_high = mafslm.stringify_alpha(alpha)
        ci_width_boot = boot_df[("abs_uplift", a_str_high)] - boot_df[("abs_uplift", a_str_low)]
        ci_width_lm = out[("abs_uplift", a_str_high)] - out[("abs_uplift", a_str_low)]
        assert ci_width_lm < ci_width_boot


def test__extract_relative_uplifts():
    results, alphas, ref_branch, model_df, column_label = _make_test_model()

    bootstrap_results = mafsb.compare_branches(
        model_df, column_label, ref_branch_label=ref_branch
    )

    # control vs treatment-a
    out = mafslm._extract_relative_uplifts(results, "control", ref_branch, alphas)

    expected_mean = (50 - 150) / 150
    assert np.isclose(out[("rel_uplift", "exp")], expected_mean, atol=0.01)
    assert np.isclose(out[("rel_uplift", "0.5")], expected_mean, atol=0.01)

    ## cross-validate with existing bootstrap implementation
    boot_df = bootstrap_results["comparative"]["control"]
    for alpha in alphas:
        a_str_low, a_str_high = mafslm.stringify_alpha(alpha)
        assert np.isclose(
            out[("rel_uplift", a_str_low)],
            boot_df[("rel_uplift", a_str_low)],
            atol=0.01,
        )
        assert np.isclose(
            out[("rel_uplift", a_str_high)],
            boot_df[("rel_uplift", a_str_high)],
            atol=0.01,
        )

    out = mafslm._extract_relative_uplifts(results, "treatment-b", ref_branch, alphas)

    expected_mean = (250 - 150) / 150
    assert np.isclose(out[("rel_uplift", "exp")], expected_mean, atol=0.01)
    assert np.isclose(out[("rel_uplift", "0.5")], expected_mean, atol=0.01)

    ## cross-validate with existing bootstrap implementation
    boot_df = bootstrap_results["comparative"]["treatment-b"]
    for alpha in alphas:
        a_str_low, a_str_high = mafslm.stringify_alpha(alpha)
        assert np.isclose(
            out[("rel_uplift", a_str_low)],
            boot_df[("rel_uplift", a_str_low)],
            atol=0.01,
        )
        assert np.isclose(
            out[("rel_uplift", a_str_high)],
            boot_df[("rel_uplift", a_str_high)],
            atol=0.01,
        )

def test__extract_relative_uplifts_covariate():
    results, alphas, ref_branch, model_df, column_label, column_label_pre = _make_test_model_covariate()

    bootstrap_results = mafsb.compare_branches(
        model_df, column_label, ref_branch_label=ref_branch
    )

    # control vs treatment-a
    out = mafslm._extract_relative_uplifts(results, "control", ref_branch, alphas)

    expected_mean = -0.1/2
    assert np.isclose(out[("rel_uplift", "exp")], expected_mean, atol=0.05)
    assert np.isclose(out[("rel_uplift", "0.5")], expected_mean, atol=0.05)

    ## compare to bootstrap implementation
    # validate the point estimates are more precise
    boot_df = bootstrap_results["comparative"]["control"]

    ###validate the widths of confidence intervals are smaller with LM approach
    for alpha in alphas:
        a_str_low, a_str_high = mafslm.stringify_alpha(alpha)
        ci_width_boot = boot_df[("rel_uplift", a_str_high)] - boot_df[("rel_uplift", a_str_low)]
        ci_width_lm = out[("rel_uplift", a_str_high)] - out[("rel_uplift", a_str_low)]
        assert ci_width_lm < ci_width_boot

    # treatment-a vs treatment-b
    out = mafslm._extract_relative_uplifts(results, "treatment-b", ref_branch, alphas)

    expected_mean = 0.1/2
    assert np.isclose(out[("rel_uplift", "exp")], expected_mean, atol=0.05)
    assert np.isclose(out[("rel_uplift", "0.5")], expected_mean, atol=0.05)

    boot_df = bootstrap_results["comparative"]["treatment-b"]
    for alpha in alphas:
        a_str_low, a_str_high = mafslm.stringify_alpha(alpha)
        ci_width_boot = boot_df[("rel_uplift", a_str_high)] - boot_df[("rel_uplift", a_str_low)]
        ci_width_lm = out[("rel_uplift", a_str_high)] - out[("rel_uplift", a_str_low)]
        assert ci_width_lm < ci_width_boot


def test_summarize_joint():
    """Validates the structure of the comparative results object,
    tests for accuracy of reported values are found above, in the
    test__extract_<>_uplifts tests"""
    _, alphas, _, model_df, column_label = _make_test_model()

    actual = mafslm.summarize_joint(
        model_df,
        column_label,
        ["control", "treatment-a", "treatment-b"],
        alphas,
        "treatment-a",
    )

    expected_index = [
        ("abs_uplift", "0.5"),
        ("abs_uplift", "exp"),
        ("abs_uplift", "0.005"),
        ("abs_uplift", "0.995"),
        ("abs_uplift", "0.025"),
        ("abs_uplift", "0.975"),
        ("rel_uplift", "0.5"),
        ("rel_uplift", "exp"),
        ("rel_uplift", "0.005"),
        ("rel_uplift", "0.995"),
        ("rel_uplift", "0.025"),
        ("rel_uplift", "0.975"),
    ]

    for branch in ["control", "treatment-b"]:
        for key in expected_index:
            assert key in actual[branch].index
        assert len(actual[branch].index) == len(expected_index)


def test__make_model_df():
    # test removal of nulls
    y = [1, 2, 3, 4, None, np.nan]
    branch = ["control", "control", "treatment", "treatment", "control", "treatment"]
    df_in = pd.DataFrame({"y": y, "branch": branch})
    df_actual = mafslm._make_model_df(df_in, "y")
    df_expected = pd.DataFrame({"branch": branch[:4], "y": [1.0, 2.0, 3.0, 4.0]})
    pd.testing.assert_frame_equal(df_actual, df_expected)

    # test removal of nulls with covariate
    y = [1, 2, 3, 4, None, np.nan]
    y2 = [1, 2, None, np.nan, 3, 4]
    branch = ["control", "control", "treatment", "treatment", "control", "treatment"]
    df_in = pd.DataFrame({"y": y, "branch": branch, "y2": y2})
    df_actual = mafslm._make_model_df(df_in, "y", covariate_col_label="y2")
    df_expected = pd.DataFrame(
        {"branch": branch[:2], "y": [1.0, 2.0], "y2": [1.0, 2.0]}
    )
    pd.testing.assert_frame_equal(df_actual, df_expected)

    # test thresholding
    y = list(range(100))
    branch = ["control"] * 100
    df_in = pd.DataFrame({"y": y, "branch": branch})
    df_actual = mafslm._make_model_df(df_in, "y", threshold_quantile=0.95)
    df_expected = pd.DataFrame(
        {
            "branch": branch,
            "y": [float(_y) for _y in y[:-5] + [np.quantile(y, 0.95)] * 5],
        }
    )
    pd.testing.assert_frame_equal(df_actual, df_expected)

    # test covariate & thresholding
    y2 = list(range(100, 200))
    branch = ["control"] * 100
    df_in = pd.DataFrame({"y": y, "branch": branch, "y2": y2})
    df_actual = mafslm._make_model_df(
        df_in, "y", covariate_col_label="y2", threshold_quantile=0.95
    )
    df_expected = pd.DataFrame(
        {
            "branch": branch,
            "y": [float(_y) for _y in y[:-5] + [np.quantile(y, 0.95)] * 5],
            "y2": [float(_y) for _y in y2[:-5] + [np.quantile(y2, 0.95)] * 5],
        }
    )
    pd.testing.assert_frame_equal(df_actual, df_expected)


def test_compare_branches_lm():
    """This is an integration type test, testing only that the
    format of the output object is correct. Functionality testing
    of specific elements of the object is covered by other tests"""
    branches = ["control", "treatment-a", "treatment-b"]
    _, alphas, _, model_df, column_label = _make_test_model()

    out = mafslm.compare_branches_lm(
        model_df, column_label, "treatment-a", alphas=alphas
    )

    assert list(out.keys()) == ["individual", "comparative"]

    # assert sorted(list(out['individual'].keys())) == branches

    for branch in branches:
        assert sorted(out["individual"][branch].index) == [
            "0.005",
            "0.025",
            "0.5",
            "0.975",
            "0.995",
            "mean",
        ]

    comparative_branches = ["control", "treatment-b"]

    assert sorted(out["comparative"].keys()) == comparative_branches

    expected_index = [
        ("abs_uplift", "0.005"),
        ("abs_uplift", "0.025"),
        ("abs_uplift", "0.5"),
        ("abs_uplift", "0.975"),
        ("abs_uplift", "0.995"),
        ("abs_uplift", "exp"),
        ("rel_uplift", "0.005"),
        ("rel_uplift", "0.025"),
        ("rel_uplift", "0.5"),
        ("rel_uplift", "0.975"),
        ("rel_uplift", "0.995"),
        ("rel_uplift", "exp"),
    ]

    for branch in comparative_branches:
        assert sorted(out["comparative"][branch].index) == expected_index