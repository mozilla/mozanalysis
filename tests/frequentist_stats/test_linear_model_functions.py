import logging
import re
from copy import deepcopy
from itertools import product

import numpy as np
import pandas as pd
import polars as pl
import pytest
import statsmodels.formula.api as smf
from polars.testing import assert_frame_equal as pl_assert_frame_equal
from statsmodels.stats.weightstats import CompareMeans

import mozanalysis.frequentist_stats.bootstrap as mafsb
import mozanalysis.frequentist_stats.linear_models as mafslm
import mozanalysis.frequentist_stats.linear_models.functions as func
import mozanalysis.types as types
from mozanalysis.frequentist_stats.linear_models.functions import (
    FailedToFitModel,
    MissingModelParameter,
)

from .helpers import test_model, test_model_covariate

logger = logging.getLogger(__name__)


def test__stringify_alpha():
    for bad_alphas in [-1, 0, 1, 2]:
        with pytest.raises(ValueError, match=r"alpha must be in \(0.002,1\)"):
            func._stringify_alpha(bad_alphas)

    alpha = 0.1
    low, high = func._stringify_alpha(alpha)
    assert low == "0.050"
    assert high == "0.950"

    alpha = 0.05
    low, high = func._stringify_alpha(alpha)
    assert low == "0.025"
    assert high == "0.975"

    alpha = 0.01
    low, high = func._stringify_alpha(alpha)
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
    control_branch_data = pd.Series(range(100))
    treatment_branch_data = pd.Series(range(100, 200))
    test_data = pd.concat([control_branch_data, treatment_branch_data])
    branches = pd.Series([*(["control"] * 100), *(["treatment"] * 100)])
    branch_list = ["control", "treatment"]

    result = mafslm.summarize_univariate(test_data, branches, [0.05])

    assert sorted(result.keys()) == branch_list
    for branch in branch_list:
        index_values = result[branch].index.values
        index_values.sort()
        assert list(index_values) == ["0.025", "0.5", "0.975", "mean"]

    # control
    ## validate against theoretical values
    mean = 49.5
    low, high = 43.74349, 55.25650
    assert np.isclose(result["control"]["mean"], mean)
    assert np.isclose(result["control"]["0.5"], mean)
    assert np.isclose(result["control"]["0.025"], low)
    assert np.isclose(result["control"]["0.975"], high)

    ## cross-validate against existing bootstrap implementation
    bootstrap_result = mafsb.bootstrap_one_branch(control_branch_data)
    assert np.isclose(result["control"]["mean"], bootstrap_result["mean"], atol=0.5)
    assert np.isclose(result["control"]["0.5"], bootstrap_result["0.5"], atol=0.5)
    assert np.isclose(result["control"]["0.025"], bootstrap_result["0.025"], atol=0.5)
    assert np.isclose(result["control"]["0.975"], bootstrap_result["0.975"], atol=0.5)

    # treatment
    ## validate against theoretical values
    mean = 149.5
    low, high = 143.74349, 155.25650
    assert np.isclose(result["treatment"]["mean"], mean)
    assert np.isclose(result["treatment"]["0.5"], mean)
    assert np.isclose(result["treatment"]["0.025"], low)
    assert np.isclose(result["treatment"]["0.975"], high)

    ## cross-validate against existing bootstrap implementation
    bootstrap_result = mafsb.bootstrap_one_branch(treatment_branch_data)
    assert np.isclose(result["treatment"]["mean"], bootstrap_result["mean"], atol=0.5)
    assert np.isclose(result["treatment"]["0.5"], bootstrap_result["0.5"], atol=0.5)
    assert np.isclose(result["treatment"]["0.025"], bootstrap_result["0.025"], atol=0.5)
    assert np.isclose(result["treatment"]["0.975"], bootstrap_result["0.975"], atol=0.5)


def test_make_formula():
    expected = "search_count ~ C(branch, Treatment(reference='control'))"
    actual = mafslm.make_formula("search_count", "control")

    assert expected == actual

    expected = "days_of_use ~ C(branch, Treatment(reference='treatment-a'))"
    actual = mafslm.make_formula("days_of_use", "treatment-a")

    assert expected == actual

    expected = "active_hours ~ C(branch, Treatment(reference='treatment-a')) + active_hours_pre"  # noqa: E501
    actual = mafslm.make_formula("active_hours", "treatment-a", "active_hours_pre")

    assert expected == actual

    for bad_target in ["search~count", "search(count", "search)count", "search_count'"]:
        with pytest.raises(
            ValueError, match=r"Target variable .* contains invalid character"
        ):
            mafslm.make_formula(bad_target, "control")

    for bad_branch in ["search~count", "search(count", "search)count", "search_count'"]:
        with pytest.raises(
            ValueError, match=r"Reference branch .* contains invalid character"
        ):
            mafslm.make_formula("search_count", bad_branch)

    for bad_covariate in [
        "search~count",
        "search(count",
        "search)count",
        "search_count'",
    ]:
        with pytest.raises(
            ValueError, match=r"Covariate .* contains invalid character"
        ):
            mafslm.make_formula("search_count", "control", bad_covariate)


def test__make_joint_output():
    out = func._make_joint_output([0.01, 0.05], types.Uplift.RELATIVE)

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
    bootstrap_results = mafsb.compare_branches(
        test_model.model_df,
        test_model.target,
        ref_branch_label=test_model.ref_branch,
    )

    # control vs treatment-a
    out = func._extract_absolute_uplifts(
        test_model.results, "control", test_model.ref_branch, test_model.alphas
    )

    expected_mean = 50 - 150
    assert np.isclose(out[("abs_uplift", "exp")], expected_mean)
    assert np.isclose(out[("abs_uplift", "0.5")], expected_mean)

    for alpha in test_model.alphas:
        a_str_low, a_str_high = func._stringify_alpha(alpha)
        low, high = CompareMeans.from_data(
            test_model.model_df.loc[
                test_model.model_df.branch == "control", test_model.target
            ],
            test_model.model_df.loc[
                test_model.model_df.branch == test_model.ref_branch, test_model.target
            ],
        ).zconfint_diff(alpha)
        assert np.isclose(out[("abs_uplift", a_str_low)], low, atol=0.1)
        assert np.isclose(out[("abs_uplift", a_str_high)], high, atol=0.1)

    ## cross-validate with existing bootstrap implementation
    boot_df = bootstrap_results["comparative"]["control"]
    for alpha in test_model.alphas:
        a_str_low, a_str_high = func._stringify_alpha(alpha)
        assert np.isclose(
            out[("abs_uplift", a_str_low)], boot_df[("abs_uplift", a_str_low)], atol=1.0
        )
        assert np.isclose(
            out[("abs_uplift", a_str_high)],
            boot_df[("abs_uplift", a_str_high)],
            atol=1.0,
        )
    # treatment-a vs treatment-b
    out = func._extract_absolute_uplifts(
        test_model.results, "treatment-b", test_model.ref_branch, test_model.alphas
    )

    expected_mean = 250 - 150
    assert np.isclose(out[("abs_uplift", "exp")], expected_mean)
    assert np.isclose(out[("abs_uplift", "0.5")], expected_mean)

    for alpha in test_model.alphas:
        a_str_low, a_str_high = func._stringify_alpha(alpha)
        low, high = CompareMeans.from_data(
            test_model.model_df.loc[
                test_model.model_df.branch == "treatment-b", test_model.target
            ],
            test_model.model_df.loc[
                test_model.model_df.branch == test_model.ref_branch, test_model.target
            ],
        ).zconfint_diff(alpha)
        assert np.isclose(out[("abs_uplift", a_str_low)], low, atol=0.1)
        assert np.isclose(out[("abs_uplift", a_str_high)], high, atol=0.1)

    ## cross-validate with existing bootstrap implementation
    boot_df = bootstrap_results["comparative"]["treatment-b"]
    for alpha in test_model.alphas:
        a_str_low, a_str_high = func._stringify_alpha(alpha)
        assert np.isclose(
            out[("abs_uplift", a_str_low)], boot_df[("abs_uplift", a_str_low)], atol=1.0
        )
        assert np.isclose(
            out[("abs_uplift", a_str_high)],
            boot_df[("abs_uplift", a_str_high)],
            atol=1.0,
        )


def test__extract_absolute_uplifts_covariate():
    bootstrap_results = mafsb.compare_branches(
        test_model_covariate.model_df,
        test_model_covariate.target,
        ref_branch_label=test_model_covariate.ref_branch,
    )

    # control vs treatment-a
    out = func._extract_absolute_uplifts(
        test_model_covariate.results,
        "control",
        test_model_covariate.ref_branch,
        test_model_covariate.alphas,
    )

    expected_mean = -0.1
    assert np.isclose(out[("abs_uplift", "exp")], expected_mean, atol=0.05)
    assert np.isclose(out[("abs_uplift", "0.5")], expected_mean, atol=0.05)

    ## compare to bootstrap implementation
    # validate the point estimates are more precise
    boot_df = bootstrap_results["comparative"]["control"]

    ###validate the widths of confidence intervals are smaller with LM approach
    for alpha in test_model_covariate.alphas:
        a_str_low, a_str_high = func._stringify_alpha(alpha)
        ci_width_boot = (
            boot_df[("abs_uplift", a_str_high)] - boot_df[("abs_uplift", a_str_low)]
        )
        ci_width_lm = out[("abs_uplift", a_str_high)] - out[("abs_uplift", a_str_low)]
        assert ci_width_lm < ci_width_boot

    # treatment-a vs treatment-b
    out = func._extract_absolute_uplifts(
        test_model_covariate.results,
        "treatment-b",
        test_model_covariate.ref_branch,
        test_model_covariate.alphas,
    )

    expected_mean = 0.1
    assert np.isclose(out[("abs_uplift", "exp")], expected_mean, atol=0.05)
    assert np.isclose(out[("abs_uplift", "0.5")], expected_mean, atol=0.05)

    boot_df = bootstrap_results["comparative"]["treatment-b"]
    for alpha in test_model_covariate.alphas:
        a_str_low, a_str_high = func._stringify_alpha(alpha)
        ci_width_boot = (
            boot_df[("abs_uplift", a_str_high)] - boot_df[("abs_uplift", a_str_low)]
        )
        ci_width_lm = out[("abs_uplift", a_str_high)] - out[("abs_uplift", a_str_low)]
        assert ci_width_lm < ci_width_boot


def test__extract_relative_uplifts():
    bootstrap_results = mafsb.compare_branches(
        test_model.model_df, test_model.target, ref_branch_label=test_model.ref_branch
    )

    # control vs treatment-a
    out = func._extract_relative_uplifts(
        test_model.results,
        "control",
        test_model.ref_branch,
        test_model.alphas,
        test_model.treatment_branches,
    )

    expected_mean = (50 - 150) / 150
    assert np.isclose(out[("rel_uplift", "exp")], expected_mean, atol=0.01)
    assert np.isclose(out[("rel_uplift", "0.5")], expected_mean, atol=0.01)

    ## cross-validate with existing bootstrap implementation
    boot_df = bootstrap_results["comparative"]["control"]
    for alpha in test_model.alphas:
        a_str_low, a_str_high = func._stringify_alpha(alpha)
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

    out = func._extract_relative_uplifts(
        test_model.results,
        "treatment-b",
        test_model.ref_branch,
        test_model.alphas,
        test_model.treatment_branches,
    )

    expected_mean = (250 - 150) / 150
    assert np.isclose(out[("rel_uplift", "exp")], expected_mean, atol=0.01)
    assert np.isclose(out[("rel_uplift", "0.5")], expected_mean, atol=0.01)

    ## cross-validate with existing bootstrap implementation
    boot_df = bootstrap_results["comparative"]["treatment-b"]
    for alpha in test_model.alphas:
        a_str_low, a_str_high = func._stringify_alpha(alpha)
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
    bootstrap_results = mafsb.compare_branches(
        test_model_covariate.model_df,
        test_model_covariate.target,
        ref_branch_label=test_model_covariate.ref_branch,
    )

    # control vs treatment-a
    out = func._extract_relative_uplifts(
        test_model_covariate.results,
        "control",
        test_model_covariate.ref_branch,
        test_model_covariate.alphas,
        test_model_covariate.treatment_branches,
    )

    expected_mean = -0.1 / 2.1
    assert np.isclose(out[("rel_uplift", "exp")], expected_mean, atol=0.05)
    assert np.isclose(out[("rel_uplift", "0.5")], expected_mean, atol=0.05)

    ## compare to bootstrap implementation
    # validate the point estimates are more precise
    boot_df = bootstrap_results["comparative"]["control"]

    ###validate the widths of confidence intervals are smaller with LM approach
    for alpha in test_model_covariate.alphas:
        a_str_low, a_str_high = func._stringify_alpha(alpha)
        ci_width_boot = (
            boot_df[("rel_uplift", a_str_high)] - boot_df[("rel_uplift", a_str_low)]
        )
        ci_width_lm = out[("rel_uplift", a_str_high)] - out[("rel_uplift", a_str_low)]
        assert ci_width_lm < ci_width_boot

    # treatment-a vs treatment-b
    out = func._extract_relative_uplifts(
        test_model_covariate.results,
        "treatment-b",
        test_model_covariate.ref_branch,
        test_model_covariate.alphas,
        test_model_covariate.treatment_branches,
    )

    expected_mean = 0.1 / 2.1
    assert np.isclose(out[("rel_uplift", "exp")], expected_mean, atol=0.05)
    assert np.isclose(out[("rel_uplift", "0.5")], expected_mean, atol=0.05)

    boot_df = bootstrap_results["comparative"]["treatment-b"]
    for alpha in test_model_covariate.alphas:
        a_str_low, a_str_high = func._stringify_alpha(alpha)
        ci_width_boot = (
            boot_df[("rel_uplift", a_str_high)] - boot_df[("rel_uplift", a_str_low)]
        )
        ci_width_lm = out[("rel_uplift", a_str_high)] - out[("rel_uplift", a_str_low)]
        assert ci_width_lm < ci_width_boot


@pytest.mark.parametrize("deallocate", [True, False])
def test_summarize_joint(deallocate):
    """Validates the structure of the comparative results object,
    tests for accuracy of reported values are found above, in the
    test__extract_<>_uplifts tests"""

    actual = mafslm.summarize_joint(
        test_model.model_df,
        test_model.target,
        test_model.alphas,
        ref_branch_label="treatment-a",
        deallocate_aggressively=deallocate,
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


@pytest.mark.parametrize("copy", [True, False])
def test_prepare_df_for_modeling(copy):
    # test removal of nulls
    y = [1, 2, 3, 4, None, np.nan]
    branch = ["control", "control", "treatment", "treatment", "control", "treatment"]
    df_in = pd.DataFrame({"y": y, "branch": branch})
    df_actual = mafslm.prepare_df_for_modeling(df_in, "y", copy=copy)
    df_expected = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "branch": branch[:4]})
    pd.testing.assert_frame_equal(df_actual, df_expected)
    # null removal does not drop rows in place
    # pd.testing.assert_frame_equal(df_in, df_expected)

    # test removal of nulls with covariate
    y = [1, 2, 3, 4, None, np.nan]
    y2 = [1, 2, None, np.nan, 3, 4]
    branch = ["control", "control", "treatment", "treatment", "control", "treatment"]
    df_in = pd.DataFrame({"y": y, "branch": branch, "y2": y2})
    df_actual = mafslm.prepare_df_for_modeling(
        df_in, "y", covariate_col="y2", copy=copy
    )
    df_expected = pd.DataFrame(
        {"y": [1.0, 2.0], "branch": branch[:2], "y2": [1.0, 2.0]}
    )
    pd.testing.assert_frame_equal(df_actual, df_expected)

    # test thresholding ints
    y = list(range(100))
    branch = ["control"] * 100
    df_in = pd.DataFrame({"y": y, "branch": branch})
    df_actual = mafslm.prepare_df_for_modeling(
        df_in, "y", threshold_quantile=0.95, copy=copy
    )
    df_expected = pd.DataFrame(
        {
            "y": [int(_y) for _y in y[:-5] + [int(np.ceil(np.quantile(y, 0.95)))] * 5],
            "branch": branch,
        }
    )
    pd.testing.assert_frame_equal(df_actual, df_expected)
    if not copy:
        pd.testing.assert_frame_equal(df_in, df_expected)

    # test thresholding floats
    y = [float(_y) for _y in (range(100))]
    branch = ["control"] * 100
    df_in = pd.DataFrame({"y": y, "branch": branch})
    df_actual = mafslm.prepare_df_for_modeling(
        df_in, "y", threshold_quantile=0.95, copy=copy
    )
    df_expected = pd.DataFrame(
        {
            "y": [float(_y) for _y in y[:-5] + [np.quantile(y, 0.95)] * 5],
            "branch": branch,
        }
    )
    pd.testing.assert_frame_equal(df_actual, df_expected)
    if not copy:
        pd.testing.assert_frame_equal(df_in, df_expected)

    # test covariate & thresholding
    y2 = [float(_y) for _y in range(100, 200)]
    branch = ["control"] * 100
    df_in = pd.DataFrame({"y": y, "branch": branch, "y2": y2})
    df_actual = mafslm.prepare_df_for_modeling(
        df_in, "y", covariate_col="y2", threshold_quantile=0.95, copy=copy
    )
    df_expected = pd.DataFrame(
        {
            "y": [float(_y) for _y in y[:-5] + [np.quantile(y, 0.95)] * 5],
            "branch": branch,
            "y2": [float(_y) for _y in y2[:-5] + [np.quantile(y2, 0.95)] * 5],
        }
    )
    pd.testing.assert_frame_equal(df_actual, df_expected)
    if not copy:
        pd.testing.assert_frame_equal(df_in, df_expected)

    # test behavior of clipping with integer dtypes
    y = list(range(100))
    branch = ["control"] * 100
    df_in = pd.DataFrame({"y": y, "branch": branch})
    df_in["y"] = df_in.y.astype(pd.Int64Dtype())
    df_actual = mafslm.prepare_df_for_modeling(
        df_in, "y", threshold_quantile=0.948, copy=copy
    )
    df_expected = pd.DataFrame(
        {
            "y": [int(_y) for _y in y[:-5] + [int(np.ceil(np.quantile(y, 0.948)))] * 5],
            "branch": branch,
        }
    )
    df_expected["y"] = df_expected.y.astype(pd.Int64Dtype())
    pd.testing.assert_frame_equal(df_actual, df_expected)


@pytest.mark.parametrize("interactive", [True, False])
@pytest.mark.parametrize("deallocate", [True, False])
def test_compare_branches_lm(interactive, deallocate):
    """This is an integration type test, testing only that the
    format of the output object is correct. Functionality testing
    of specific elements of the object is covered by other tests"""
    branches = ["control", "treatment-a", "treatment-b"]

    out = mafslm.compare_branches_lm(
        test_model.model_df,
        test_model.target,
        "treatment-a",
        alphas=test_model.alphas,
        interactive=interactive,
        deallocate_aggressively=deallocate,
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


def test_compare_branches_lm_fallback(caplog):
    # tests whether the results when called with a missing covariate equal the
    # results when called without a covariate. Also tests that a log was emitted.
    msg = "Covariate fake_covariate not found, falling back to unadjusted inferences"

    assert msg not in caplog.text

    actual = mafslm.compare_branches_lm(
        test_model_covariate.model_df,
        test_model_covariate.target,
        test_model_covariate.ref_branch,
        "fake_covariate",
    )

    expected = mafslm.compare_branches_lm(
        test_model_covariate.model_df,
        test_model_covariate.target,
        test_model_covariate.ref_branch,
    )

    for branch in test_model_covariate.treatment_branches:
        pd.testing.assert_series_equal(
            actual["comparative"][branch], expected["comparative"][branch]
        )

    assert msg in caplog.text


def test_fit_model():
    actual_results = mafslm.fit_model(
        test_model.model_df,
        test_model.target,
        test_model.ref_branch,
        test_model.treatment_branches,
    )

    pd.testing.assert_series_equal(actual_results.params, test_model.results.params)

    alpha = 0.05
    pd.testing.assert_frame_equal(
        actual_results.conf_int(alpha), test_model.results.conf_int(alpha)
    )


def test_fit_model_covariate():
    actual_results = mafslm.fit_model(
        test_model_covariate.model_df,
        test_model_covariate.target,
        test_model_covariate.ref_branch,
        test_model_covariate.treatment_branches,
        test_model_covariate.covariate,
    )

    pd.testing.assert_series_equal(
        actual_results.params, test_model_covariate.results.params
    )

    alpha = 0.05
    pd.testing.assert_frame_equal(
        actual_results.conf_int(alpha), test_model_covariate.results.conf_int(alpha)
    )


def test_fit_model_covariate_robust_to_bad_covariate():
    model_df = test_model_covariate.model_df.copy()
    model_df.loc[:, test_model_covariate.covariate] = [0] * model_df.shape[0]

    expected_results = smf.ols(test_model.formula, model_df).fit()

    with pytest.warns(Warning, match="Unexpectedly fell back to unadjusted inferences"):
        actual_results = mafslm.fit_model(
            model_df,
            test_model_covariate.target,
            test_model_covariate.ref_branch,
            test_model_covariate.treatment_branches,
            test_model_covariate.covariate,
        )

    pd.testing.assert_series_equal(actual_results.params, expected_results.params)

    alpha = 0.05
    pd.testing.assert_frame_equal(
        actual_results.conf_int(alpha), expected_results.conf_int(alpha)
    )


def test_fit_model_covariate_fails_on_bad_data():
    model_df = test_model_covariate.model_df.copy()
    model_df.loc[:, test_model_covariate.target] = [0] * model_df.shape[0]

    with pytest.raises(
        FailedToFitModel,
        match="Failed to fit model for target search_count using covariate search_count_pre",  # noqa: E501
    ):
        mafslm.fit_model(
            model_df,
            test_model_covariate.target,
            test_model_covariate.ref_branch,
            test_model_covariate.treatment_branches,
            test_model_covariate.covariate,
        )


def test_fit_model_covariate_fails_on_bad_branch():
    model_df = test_model_covariate.model_df.copy()

    model_df.loc[:, "branch"] = ["treatment-a"] * model_df.shape[0]

    with pytest.raises(
        MissingModelParameter, match="Effect for branch control not found in model!"
    ):
        mafslm.fit_model(
            model_df,
            test_model_covariate.target,
            test_model_covariate.ref_branch,
            test_model_covariate.treatment_branches,
            test_model_covariate.covariate,
        )


def test__create_datagrid():
    # without covariate
    branches = test_model.treatment_branches + [test_model.ref_branch]
    newdata = func._create_datagrid(deepcopy(test_model.results), branches)

    expected_data = []
    for branch in branches:
        expected_data.append(
            {
                "branch": branch,
            }
        )
    expected = pl.DataFrame(expected_data)
    pl_assert_frame_equal(newdata.select("branch"), expected)

    # with covariate
    branches = test_model_covariate.treatment_branches + [
        test_model_covariate.ref_branch
    ]
    newdata = func._create_datagrid(
        test_model_covariate.results, branches, test_model_covariate.covariate
    )
    expected_data = []
    for branch in branches:
        for q in test_model_covariate.model_df[test_model_covariate.covariate].quantile(
            np.arange(0, 1, 0.0001)
        ):
            expected_data.append({"branch": branch, test_model_covariate.covariate: q})
    expected = pl.DataFrame(expected_data)

    pl_assert_frame_equal(
        newdata.select("branch", test_model_covariate.covariate).sort(
            "branch", test_model_covariate.covariate
        ),
        expected.sort("branch"),
    )

    # handle pandas types
    branches = test_model_covariate.treatment_branches + [
        test_model_covariate.ref_branch
    ]
    model_df = test_model_covariate.model_df.copy()

    model_df["search_count"] = model_df.search_count.astype(int).astype(pd.Int64Dtype())
    model_df[test_model_covariate.covariate] = (
        model_df[test_model_covariate.covariate].astype(int).astype(pd.Int64Dtype())
    )

    results = smf.ols(test_model.formula, model_df).fit()
    # should not throw, even though types are pandas types
    func._create_datagrid(results, branches, test_model_covariate.covariate)


def test_relative_inferences_with_without_datagrid():
    from marginaleffects import avg_comparisons

    branches = test_model.treatment_branches + [test_model.ref_branch]
    datagrid = func._create_datagrid(deepcopy(test_model.results), branches)

    for branch in test_model.treatment_branches:
        expected = avg_comparisons(
            test_model.results,
            variables={"branch": [test_model.ref_branch, branch]},
            comparison="lnratioavg",
            transform=np.exp,
            conf_level=1 - 0.05,
        )

        assert expected.shape == (
            1,
            7,
        ), "avg_comparisons result object not shaped as expected"

        actual = avg_comparisons(
            test_model.results,
            variables={"branch": [test_model.ref_branch, branch]},
            comparison="lnratioavg",
            transform=np.exp,
            conf_level=1 - 0.05,
            newdata=datagrid,
        )

        assert actual.shape == (
            1,
            7,
        ), "avg_comparisons result object not shaped as expected"

        assert np.isclose(expected["conf_low"][0], actual["conf_low"][0])
        assert np.isclose(expected["estimate"][0], actual["estimate"][0])
        assert np.isclose(expected["conf_high"][0], actual["conf_high"][0])


def test_relative_inferences_with_without_datagrid_covariate():
    from marginaleffects import avg_comparisons

    branches = test_model_covariate.treatment_branches + [
        test_model_covariate.ref_branch
    ]
    datagrid = func._create_datagrid(deepcopy(test_model_covariate.results), branches)

    for branch in test_model_covariate.treatment_branches:
        expected = avg_comparisons(
            test_model_covariate.results,
            variables={"branch": [test_model_covariate.ref_branch, branch]},
            comparison="lnratioavg",
            transform=np.exp,
            conf_level=1 - 0.05,
        )

        assert expected.shape == (
            1,
            7,
        ), "avg_comparisons result object not shaped as expected"

        actual = avg_comparisons(
            test_model_covariate.results,
            variables={"branch": [test_model_covariate.ref_branch, branch]},
            comparison="lnratioavg",
            transform=np.exp,
            conf_level=1 - 0.05,
            newdata=datagrid,
        )

        assert actual.shape == (
            1,
            7,
        ), "avg_comparisons result object not shaped as expected"

        assert np.isclose(expected["conf_low"][0], actual["conf_low"][0])
        assert np.isclose(expected["estimate"][0], actual["estimate"][0])
        assert np.isclose(expected["conf_high"][0], actual["conf_high"][0])


def test__validate_parameters():
    with pytest.raises(ValueError, match="Target metric fake_var not found in data"):
        func._validate_parameters(
            test_model.model_df, "fake_var", test_model.ref_branch, None, None, None
        )

    df = test_model.model_df.copy()
    df["search_count"] = 1
    with pytest.raises(
        func.UnableToAnalyze, match="Metric search_count has no variation"
    ):
        func._validate_parameters(
            df, "search_count", test_model.ref_branch, None, None, None
        )

    with pytest.raises(
        ValueError, match="No data from reference branch fake_branch found"
    ):
        func._validate_parameters(
            test_model.model_df, test_model.target, "fake_branch", None, None, None
        )

    msg = re.escape("Threshold quantile must be in (0,1]")
    with pytest.raises(ValueError, match=msg):
        func._validate_parameters(
            test_model.model_df,
            test_model.target,
            test_model.ref_branch,
            None,
            -1,
            None,
        )

    with pytest.raises(ValueError, match=msg):
        func._validate_parameters(
            test_model.model_df,
            test_model.target,
            test_model.ref_branch,
            None,
            0,
            None,
        )

    with pytest.raises(ValueError, match=msg):
        func._validate_parameters(
            test_model.model_df,
            test_model.target,
            test_model.ref_branch,
            None,
            1.00000001,
            None,
        )

    msg = re.escape("alpha must be in (0.002,1)")
    with pytest.raises(ValueError, match=msg):
        func._validate_parameters(
            test_model.model_df,
            test_model.target,
            test_model.ref_branch,
            None,
            None,
            [0.001],
        )

    with pytest.raises(ValueError, match=msg):
        func._validate_parameters(
            test_model.model_df,
            test_model.target,
            test_model.ref_branch,
            None,
            None,
            [1.001],
        )

    with pytest.raises(ValueError, match=msg):
        func._validate_parameters(
            test_model.model_df,
            test_model.target,
            test_model.ref_branch,
            None,
            None,
            [0.05, 0.1, 0.01, 0.000001],
        )

    with pytest.raises(
        ValueError,
        match=f"Covariate {test_model_covariate.target} must be different than target {test_model_covariate.target}",  # noqa: E501
    ):
        func._validate_parameters(
            test_model_covariate.model_df,
            test_model_covariate.target,
            test_model_covariate.ref_branch,
            test_model_covariate.target,
            None,
            None,
        )

    with pytest.raises(
        func.CovariateNotFound, match="Covariate fake_covariate not found in data"
    ):
        func._validate_parameters(
            test_model_covariate.model_df,
            test_model_covariate.target,
            test_model_covariate.ref_branch,
            "fake_covariate",
            None,
            None,
        )


def test__make_empty_compare_branches_output():
    df = pd.DataFrame({"branch": ["control", "treatment-a", "treatment-b"]})
    str_quantiles = ["0.5", "exp", "0.025", "0.975"]
    index = pd.MultiIndex.from_tuples(
        product(["abs_uplift", "rel_uplift"], str_quantiles)
    )
    estimates = pd.Series(index=index)

    treatment_branches = ["treatment-a", "treatment-b"]

    actual = func._make_empty_compare_branches_output(
        "control", df, [0.05], treatment_branches
    )

    for branch_df in actual["comparative"].values():
        pd.testing.assert_series_equal(estimates.sort_index(), branch_df.sort_index())
