# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from marginaleffects import avg_comparisons
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.stats.weightstats import DescrStatsW

from mozanalysis.utils import filter_outliers


def stringify_alpha(alpha: float) -> tuple[str, str]:
    """Converts a floating point alpha-level to the string
    labels of the endpoint of a confidence interval.
    E.g., 0.05 -> '0.025', '0.975'"""
    if alpha < 0.002 or alpha >= 1:
        raise ValueError("alpha must be in (0.002,1)")
    return f"{alpha/2:0.3f}", f"{1-alpha/2:0.3f}"


def summarize_one_branch(branch_data: pd.Series, alphas: list[float]) -> pd.Series:
    """Inferences (point estimate and confidence intervals) for
    the mean of a single branch's data. Constructs confidence
    intervals from central limit theory (uses the t-distribution)

    Parameters:
    - branch_data (pd.Series): the vector of observations from a
    single branch.
    - alphas (list[float]): the desired confidence levels

    Returns:
    - result (pd.Series): the resulting inferences on the mean. Has
    the following elements in its index (assuming `alphas=[0.01,0.05]`):
      - 'mean': the point estimate of the mean
      - '0.5': also the point estimate, included for backwards compatibility
    with prior bootstrap implementations
      - '0.005': the lower bound of the 99% confidence interval
      - '0.025': the lower bound of the 95% confidence interval
      - '0.975': the upper bound of the 95% confidence interval
      - '0.995': the upper bound of the 99% confidence interval
    """
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


def _infer_branch_list(branches: pd.Series, branch_list: list[str] | None) -> list[str]:
    """Determine the list of distinct branches. Used so that `summarize_univariate`
    and `summarize_joint` can take an optional `branch_list` parameter."""
    if branch_list:
        return branch_list

    return list(branches.unique())


def summarize_univariate(
    data: pd.Series,
    branches: pd.Series,
    alphas: list[float],
    branch_list: list[str] | None = None,
) -> dict[str, pd.Series]:
    """Univariate inferences (point estimates and confidence intervals) for the
    mean of each branch's data.

    Parameters:
    - data (pd.Series): the vector of observations
    - branches (pd.Series): the vector of branch names. Expected to be the same length
    as `data`
    - branch_list (list[str]): the list of branches to perform inferences on
    - alphas (list[float]): the desired confidence levels

    Returns:
    - out (dict[str, pd.Series]): a dictionary keyed by branch label with a pandas
    series as value. The format of the series is described above in
    `summarize_one_branch`
    """

    branch_list = _infer_branch_list(branches, branch_list)

    return {
        b: summarize_one_branch(data.loc[(branches == b).values], alphas)
        for b in branch_list
    }


def _make_formula(target: str, ref_branch: str, covariate: str | None = None) -> str:
    """Makes a formula which defines the model to build. Includes terms for
    treatment branch and, optionally, a main effect for a (generally pre-experiment)
    covariate.

    Ex: given `target` of 'Y' and `ref_branch` of control, the formula will be:
    `Y ~ C(branch, Treatment(reference='control'))`. That is, we want to predict `Y`
    with treatment branch (`branch`) treated as a categorical predictor with
    reference level of `'control'`.

    This builds the following linear model (assuming 2 other treatment branches, t1
    and t2): `Y_i = \beta_0 + \beta_1*I(branch == 't1') + \beta_2*I(branch == 't2')`
    where `I` is the indicator function.

    Inferences on `\beta_1` are inferences of the average treatment effect (ATE) of
    branch t1. That is, the confidence interval for `\beta_1` is the confidence
    interval for the ATE of branch t1. Similarly for t2 and `\beta_2`.

    We can incorporate a single (generally pre-experiment) covariate. Adding a covariate
    of `Ypre` to the above, we'll build the following formula:
    `Y ~ C(branch, Treatment(reference='control')) + Ypre`
    which will fit the following linear model:

    `Y_i = \beta_0 + \beta_1*I(branch == 't1') + \beta_2*I(branch == 't2')
    + \beta_3* Ypre_i`

    For now, we elect to not include branch by covariate interaction terms and instead
    to perform inferences only on the population-level ATE.

    Parameters:
    - target (str): the variable of interest.
    - ref_branch (str): the name of the reference branch
    - covariate (Optional[str]): the name of a covariate to include in the model.

    Returns:
    - formula (str): the R-style formula, to be passed to statsmodels's formula API.

    """
    pattern = re.compile(r"(\(|\)|\~|\')")
    if pattern.findall(target):
        raise ValueError(f"Target variable {target} contains invalid character")
    if pattern.findall(ref_branch):
        raise ValueError(f"Reference branch {ref_branch} contains invalid character")
    if covariate is not None and pattern.findall(covariate):
        raise ValueError(f"Covariate {covariate} contains invalid character")

    formula = f"{target} ~ C(branch, Treatment(reference='{ref_branch}'))"
    if covariate is not None:
        formula += f" + {covariate}"

    return formula


def _make_joint_output(alphas: list[float], uplift_type: str) -> pd.Series:
    """Constructs an empty pandas series to hold comparative results. The series
    will be multiindexed for backwards compatability with the bootstrap results.

    Parameters:
    - alphas (list[float]): the desired confidence levels
    - uplift_type (str): either `abs_uplift` or `rel_uplift` for inferences on the
    absolute and relative differences between branches, respectively.

    Returns:
    - series (pd.Series): the empty series. Will have keys of (uplift_type, '0.5')
    and (uplift_type, 'exp'), as well as 2 keys, one for each alpha. For more info
    on keys, see `summarize_one_branch` and `stringify_alpha` above.
    """
    str_quantiles = ["0.5", "exp"]
    for alpha in alphas:
        str_quantiles.extend(stringify_alpha(alpha))
    str_quantiles.sort()
    index = pd.MultiIndex.from_tuples([(uplift_type, q) for q in str_quantiles])
    series = pd.Series(index=index, dtype="float")

    return series


def _extract_absolute_uplifts(
    results: RegressionResults, branch: str, ref_branch: str, alphas: list[float]
) -> pd.Series:
    """Extracts inferences on absolute differences between branches from a fitted
    linear model. These are simply the point estimates and confidence intervals of the
    appropriate term in the model.

    Parameters:
    - results (RegressionResults): the fitted model.
    - branch (str): the name of the branch inferences are desired on. Must have been
    present in the training data when the model was fit.
    - ref_branch (str): the name of the reference branch.
    - alphas (list[float]): the disired confidence levels.

    Returns:
    - output (pd.Series): the set of inferences. See `_make_joint_output`.
    """
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
    """Extracts inferences on relative differences between branches from a fitted
    linear model. Unlike absolute differences, these are not simply existing parameters.
    We desire inferences on (T-C)/C = T/C-1. Since 1 is a constant, we desire inferences
    on T/C. We can accomplish this through inferences on exp(ln(T/C)) =
    exp(ln(T) - ln(C)). The ln(T)-ln(C) contrast is available in
    `marginaleffects.avg_comparisons` as `lnratioavg`.

    See [this stats.stackexchange](https://stats.stackexchange.com/a/646462/82977)
    answer for more information.

    Parameters:
    - results (RegressionResults): the fitted model.
    - branch (str): the name of the branch inferences are desired on. Must have been
    present in the training data when the model was fit.
    - ref_branch (str): the name of the reference branch.
    - alphas (list[float]): the disired confidence levels.

    Returns:
    - output (pd.Series): the set of inferences. See `_make_joint_output`.
    """
    output = _make_joint_output(alphas, "rel_uplift")

    for alpha in alphas:
        # inferences on branch/ref_branch
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
        # subtract 1 b/c branch/reference - 1 = (branch - reference)/reference
        output.loc[("rel_uplift", low_str)] = ac["conf_low"][0] - 1
        output.loc[("rel_uplift", high_str)] = ac["conf_high"][0] - 1
        output.loc[("rel_uplift", "0.5")] = ac["estimate"][0] - 1
        output.loc[("rel_uplift", "exp")] = ac["estimate"][0] - 1

    return output


def fit_model(
    df: pd.DataFrame, target: str, ref_branch: str, covariate: str | None = None
) -> RegressionResults | None:
    """Fits a linear regression model to `df` using the provided formula. See
    `_make_formula` for a more in-depth discussion on the model structure.

    Parameters:
    - formula (str): the model formula (such as the output of `_make_formula`)
    - df (pd.DataFrame): the model data (such as the output of `make_model_df`)

    Returns:
    - results (RegressionResults): the fitted model results object.
    """
    formula = _make_formula(target, ref_branch, covariate)
    try:
        results = smf.ols(formula, df).fit(method="qr")
    except np.linalg.LinAlgError as lae:
        if covariate is None:
            # nothing we can do about this
            raise lae
        else:
            # maybe covariate is bad (e.g., pre-treatment covariate in
            # onboarding experiment is always zero), try falling back to
            # unadjusted inferences
            formula = _make_formula(target, ref_branch, None)
            results = smf.ols(formula, df).fit(method="qr")
            warnings.warn("Fell back to unadjusted inferences", stacklevel=1)

    if not np.isfinite(results.llf):
        raise Exception("Error fitting model")

    return results


def summarize_joint(
    df: pd.DataFrame,
    col_label: str,
    alphas: list[float],
    branch_list: list[str] | None = None,
    ref_branch_label: str = "control",
    covariate_col_label: str | None = None,
) -> dict[str, pd.Series]:
    """The primary entrypoint for linear model based inferences on comparisons
    of treatment branches. Computes absolute and relative differences between
    each treatment branch relative to the reference branch (point estimates and
    confidence intervals).

    Absolute differences are the average treatment effect (ATE): T - C, for branch
    T and reference branch C.

    Relative differences are (T-C)/C for branch T and reference branch C.

    When `covariate_col_label` is passed, the covariate is included into the model and
    affects inferences on the branch differences. The current recommendation is to
    use the pre-experiment version of the metric under measurement (col_label). E.g.,
    if studying active_hours, incorporate average active_hours during the week or month
    prior to the experiment.

    Parameters:
    - df (pd.DataFrame): a cleaned set of data, ready for modeling (such as the output
    of `make_model_df`)
    - col_label (str): the target variable for which inferences are desired.
    - branch_list (list[str]): the set of all branches (treatments and controls) in the
    experiment.
    - alphas (list[str]): the desired confidence levels.
    - ref_branch_label (str): the name of the reference branch (e.g., 'control')
    - covariate_col_label (Optional[str]): the name of a covariate to include in the
    model.

    Returns:
    - output (dict[str, pd.Series]): a dictionary keyed by (non-reference) branch,
    containing the comparative results of that branch against the reference branch.

    """

    branch_list = _infer_branch_list(df.branch, branch_list)

    treatment_branches = [b for b in branch_list if b != ref_branch_label]

    results = fit_model(df, col_label, ref_branch_label, covariate_col_label)

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


def make_model_df(
    df: pd.DataFrame,
    col_label: str,
    covariate_col_label: str | None = None,
    threshold_quantile: float | None = None,
) -> pd.DataFrame:
    """Prepares a dataset for modeling. Removes nulls from the response variable
    (col_label). Optionally, adds a similarly cleaned covariate column.
    Optionally, applies outlier filtering (to both the response variable and
    the covariate, if one exists).

    Parameters:
    - df (pd.DataFrame): the raw, client-level data. Must have a column "branch".
    - col_label (str): the label of the response variable. Must be a column in df. Only
    non-null values are modeled.
    - covariate_col_label (Optional[str]): the name of a covariate column to include.
    Must be a column in df. If included, only non-null values are modeled.
    - threshold_quantile (Optional[float]): the outlier threshold. See
    `filter_outliers`.

    Returns:
    - model_df (pd.DataFrame): the cleaned data, ready for modeling.
    """

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
) -> dict[str, dict[str, pd.Series]]:
    """Performs individual and comparative inferences on branches using standard
    central limit theory (t-tests and linear regressions). Can incorporate a covariate
    (`covariate_col_label`) to increase precision of comparative inferences.

    See `summarize_univariate` and `summarize_joint` for more information.

    Parameters:
    - df (pd.DataFrame): the raw, client-level data. Must have a column "branch".
    - col_label (str): the label of the response variable. Must be a column in df. Only
    non-null values are modeled.
    - ref_branch_label (str): the label of the reference branch (e.g., "control").
    - covariate_col_label (Optional[str]): the name of a covariate column to adjust for.
    If not passed, unadjusted inferences are provided.
    - threshold_quantile (Optional[float]): the outlier threshold. See `filter_outliers`
    - alphas (list[float]): the desired confidence levels. Defaults to [0.01, 0.05]
    (99% and 95% confidence) if not passed.

    Returns:
    - out (dict[str, dict[str, pd.Series]]): the results. Has keys `individual` for
    univariate inferences and `comparative` for joint. See `summarize_univariate`
    and `summarize_joint` for more information on the output structure.

    """

    if alphas is None:
        alphas = [0.01, 0.05]

    model_df = make_model_df(df, col_label, covariate_col_label, threshold_quantile)

    branch_list = _infer_branch_list(model_df.branch, None)

    return {
        "individual": summarize_univariate(
            model_df[col_label], model_df.branch, alphas, branch_list=branch_list
        ),
        "comparative": summarize_joint(
            model_df,
            col_label,
            alphas,
            ref_branch_label=ref_branch_label,
            branch_list=branch_list,
            covariate_col_label=covariate_col_label,
        ),
    }
