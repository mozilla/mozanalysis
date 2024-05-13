# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from marginaleffects import avg_comparisons
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.stats.weightstats import DescrStatsW

from mozanalysis.utils import filter_outliers


def stringify_alpha(alpha: float) -> tuple[str, str]:
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0,1)")
    return f"{alpha/2:0.3f}", f"{1-alpha/2:0.3f}"


def summarize_one_branch(branch_data: pd.Series, alphas: list[float]) -> pd.Series:
    str_quantiles = ["0.5"]
    for alpha in alphas:
        str_quantiles.extend(stringify_alpha(alpha))
    res = pd.Series(index=sorted(str_quantiles) + ["mean"], dtype="float")
    dsw = DescrStatsW(branch_data)
    mean = dsw.mean
    res["0.5"] = mean  # backwards compatibility
    res["mean"] = mean
    for alpha in alphas:
        low, high = dsw.tconfint_mean(alpha)
        low_str, high_str = stringify_alpha(alpha)
        res[low_str] = low
        res[high_str] = high
    return res


def summarize_univariate(
    data: pd.Series, branches: pd.Series, branch_list: list[str], alphas: list[float]
) -> dict[str, pd.Series]:
    return {
        b: summarize_one_branch(data.loc[(branches == b).values], alphas)
        for b in branch_list
    }


def _make_formula(target: str, ref_branch: str, covariate: str | None = None) -> str:
    formula = f"{target} ~ C(branch, Treatment(reference='{ref_branch}'))"
    if covariate is not None:
        formula += f" + {covariate}"

    return formula


def _make_joint_output(alphas: list[float], uplift_type: str) -> pd.Series:
    str_quantiles = ["0.5"]
    for alpha in alphas:
        str_quantiles.extend(stringify_alpha(alpha))
    str_quantiles.sort()
    index = pd.MultiIndex.from_tuples(
        [(uplift_type, q) for q in str_quantiles + ["exp"]]
    )
    series = pd.Series(index=index, dtype="float")

    return series


def _extract_absolute_uplifts(
    results: RegressionResults, branch: str, ref_branch: str, alphas: list[float]
) -> pd.Series:
    output = _make_joint_output(alphas, "abs_uplift")

    parameter_name = f"C(branch, Treatment(reference='{ref_branch}'))[T.{branch}]"

    output.loc[("abs_uplift", "0.5")] = results.params[parameter_name]
    output.loc[("abs_uplift", "exp")] = results.params[parameter_name]

    for alpha in alphas:
        lower, upper = results.conf_int(alpha=alpha).loc[parameter_name]
        low_str, high_str = stringify_alpha(alpha)
        output.loc[("abs_uplift", low_str)] = lower
        output.loc[("abs_uplift", high_str)] = upper

    return output


def _extract_relative_uplifts(
    results: RegressionResults, branch: str, ref_branch: str, alphas: list[str]
) -> pd.Series:
    output = _make_joint_output(alphas, "rel_uplift")

    for alpha in alphas:
        ac = avg_comparisons(
            results,
            variables={"branch": [ref_branch, branch]},
            comparison="lnratioavg",
            transform=np.exp,
            conf_level=1 - alpha,
        )

        assert ac.shape == (
            1,
            7,
        ), "avg_comparisons result object not shaped as expected"

        low_str, high_str = stringify_alpha(alpha)
        output.loc[("rel_uplift", low_str)] = ac["conf_low"][0] - 1
        output.loc[("rel_uplift", high_str)] = ac["conf_high"][0] - 1
        output.loc[("rel_uplift", "0.5")] = ac["estimate"][0] - 1
        output.loc[("rel_uplift", "exp")] = ac["estimate"][0] - 1

    return output


def fit_model(formula: str, df: pd.DataFrame) -> RegressionResults:
    results = smf.ols(formula, df).fit()
    return results


def summarize_joint(
    df: pd.DataFrame,
    col_label: str,
    branch_list: list[str],
    alphas: list[float],
    ref_branch_label="control",
    covariate_col_label: str | None = None,
):
    treatment_branches = [b for b in branch_list if b != ref_branch_label]

    formula = _make_formula(col_label, ref_branch_label, covariate_col_label)

    results = fit_model(formula, df)

    output = {}

    for branch in treatment_branches:
        rel_uplifts = _extract_absolute_uplifts(
            results, branch, ref_branch_label, alphas
        )

        abs_uplifts = _extract_relative_uplifts(
            results, branch, ref_branch_label, alphas
        )

        output[branch] = pd.concat([rel_uplifts, abs_uplifts])

    return output


def _make_model_df(
    df: pd.DataFrame,
    col_label: str,
    covariate_col_label: str | None = None,
    threshold_quantile: float | None = None,
) -> pd.DataFrame:

    indexer = ~df[col_label].isna()
    if covariate_col_label is not None:
        indexer &= ~df[covariate_col_label].isna()
    if threshold_quantile is not None:
        x = filter_outliers(df.loc[indexer, col_label], threshold_quantile)
    else:
        x = df.loc[indexer, col_label]

    model_df = pd.DataFrame(
        {"branch": df.loc[indexer, "branch"], col_label: x.astype(float)}
    )

    if covariate_col_label is not None:
        if threshold_quantile is not None:
            x_pre = filter_outliers(
                df.loc[indexer, covariate_col_label], threshold_quantile
            )
        else:
            x_pre = df.loc[indexer, covariate_col_label]
        model_df.loc[:, covariate_col_label] = x_pre.astype(float)

    return model_df


def compare_branches_lm(
    df: pd.DataFrame,
    col_label: str,
    ref_branch_label="control",
    covariate_col_label: str | None = None,
    threshold_quantile: float | None = None,
    alphas: list[float] | None = None,
):

    if alphas is None:
        alphas = [0.01, 0.05]

    model_df = _make_model_df(df, col_label, covariate_col_label, threshold_quantile)

    branch_list = model_df.branch.unique()

    return {
        "individual": summarize_univariate(
            model_df[col_label], model_df.branch, branch_list, alphas
        ),
        "comparative": summarize_joint(
            model_df,
            col_label,
            branch_list,
            alphas,
            ref_branch_label,
            covariate_col_label,
        ),
    }
