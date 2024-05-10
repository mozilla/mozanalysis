# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from marginaleffects import avg_comparisons
from statsmodels.stats.weightstats import DescrStatsW

from mozanalysis.utils import filter_outliers


def stringify_alpha(alpha: float) -> tuple[str, str]:
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0,1)")
    return f"{alpha/2:0.3f}", f"{1-alpha/2:0.3f}"


def summarize_one_branch(branch_data: pd.Series, alphas: list[float]):
    str_quantiles = ["0.5"]
    for alpha in alphas:
        str_quantiles.extend(stringify_alpha(alpha))
    res = pd.Series(index=sorted(str_quantiles) + ["mean"], dtype="float")
    dsw = DescrStatsW(branch_data)
    mean = dsw.mean
    res["0.5"] = mean  # backwards compatibility
    res["exp"] = mean
    for alpha in alphas:
        low, high = dsw.tconfint_mean(alpha)
        low_str, high_str = stringify_alpha(alpha)
        res[low_str] = low
        res[high_str] = high
    return res


def summarize_univariate(
    data: pd.Series, branches: pd.Series, branch_list: list[str], alphas: list[float]
):
    return {b: summarize_one_branch(data[branches == b], alphas) for b in branch_list}


def summarize_joint(
    df: pd.DataFrame,
    col_label: str,
    branch_list: list[str],
    alphas: list[float],
    ref_branch_label="control",
    pretreatment_col_label: str | None = None,
):
    treatment_branches = [b for b in branch_list if b != ref_branch_label]
    formula = f"{col_label} ~ C(branch, Treatment(reference='{ref_branch_label}'))"
    if pretreatment_col_label is not None:
        formula += f" + {pretreatment_col_label}"

    model = smf.ols(formula, df).fit()
    branch_parameters = {
        branch: f"C(branch, Treatment(reference='{ref_branch_label}'))[T.{branch}]"
        for branch in treatment_branches
    }
    str_quantiles = ["0.5"]
    for alpha in alphas:
        str_quantiles.extend(stringify_alpha(alpha))
    str_quantiles.sort()
    index = pd.MultiIndex.from_tuples(
        [("rel_uplift", q) for q in str_quantiles + ["exp"]]
        + [("abs_uplift", q) for q in str_quantiles + ["exp"]]
        # + [("max_abs_diff", "0.95"), ("prob_win",)]
    )
    res = pd.Series(index=index, dtype="float")

    output = {branch: res.copy() for branch in treatment_branches}
    for branch, parameter_name in branch_parameters.items():
        output[branch].loc[("abs_uplift", "0.5")] = model.params[parameter_name]
        output[branch].loc[("abs_uplift", "exp")] = model.params[parameter_name]

    for alpha in alphas:
        for branch, parameter_name in branch_parameters.items():
            lower, upper = model.conf_int(alpha=alpha).loc[parameter_name]
            low_str, high_str = stringify_alpha(alpha)
            output[branch].loc[("abs_uplift", low_str)] = lower
            output[branch].loc[("abs_uplift", high_str)] = upper

    for alpha in alphas:
        for branch in treatment_branches:
            ac = avg_comparisons(
                model,
                variables={"branch": [ref_branch_label, branch]},
                comparison="lnratioavg",
                transform=np.exp,
                conf_level=1 - alpha,
            )
            assert ac.shape == (1, 7), (
                "avg_comparisons result object not shaped" " as expected"
            )
            low_str, high_str = stringify_alpha(alpha)
            output[branch].loc[("rel_uplift", low_str)] = ac["conf_low"][0] - 1
            output[branch].loc[("rel_uplift", high_str)] = ac["conf_high"][0] - 1
            output[branch].loc[("rel_uplift", "0.5")] = ac["estimate"][0] - 1
            output[branch].loc[("rel_uplift", "exp")] = ac["estimate"][0] - 1

    return output


def compare_branches_lm(
    df: pd.DataFrame,
    col_label: str,
    ref_branch_label="control",
    pretreatment_col_label: str | None = None,
    threshold_quantile: float | None = None,
    alphas: list[float] | None = None,
):

    if alphas is None:
        alphas = [0.01, 0.05]
    indexer = ~df[col_label].isna()
    if pretreatment_col_label is not None:
        indexer &= ~df[pretreatment_col_label].isna()
    if threshold_quantile is not None:
        x = filter_outliers(df.loc[indexer, col_label], threshold_quantile)
    else:
        x = df.loc[indexer, col_label]

    model_df = pd.DataFrame(
        {"branch": df.loc[indexer, "branch"], col_label: x.astype(float)}
    )
    branch_list = df.branch.unique()

    if pretreatment_col_label is not None:
        if threshold_quantile is not None:
            x_pre = filter_outliers(
                df.loc[indexer, pretreatment_col_label], threshold_quantile
            )
        else:
            x_pre = df.loc[indexer, pretreatment_col_label]
        model_df.loc[:, pretreatment_col_label] = x_pre.astype(float)

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
            pretreatment_col_label,
        ),
    }
