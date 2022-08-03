# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from statsmodels.stats.power import TTestPower

def sample_size_calc(df, metrics_list, rel_effect_size = .01, alpha = .05, power = .90, solver = None):

    """
    Perform sample size calculation for an experiment.

    Args:
        df: A pandas DataFrame of queried historical data.
        metrics_list (list of mozanalysis.metrics.Metric): List of metrics
            used to construct the results df from HistoricalTarget. The names
            of these metrics are used to return results for sample size 
            calculation for each
        rel_effect_size (float, default .01): Percent change in metrics expected as a result 
            of the experiment treatment
        alpha (float, default .05): Significance level for the experiment.
        power (float, default .90): Probability of detecting an effect, when a significant effect
            exists.
        solver (class with solve_power method, default None): Argument for users to pass a 
            user-defined sample size solver; defaults to None, in which case 
            statsmodels.stats.power.TTestPower.solve_power is used

    Returns a sorted dictionary:
        Keys in the dictionary are the metrics column names from the DataFrame; values
        are the required sample size to achieve the desired power for that metric.
        The dictionary is sorted by required sample size, in descending order.        
    """

    if not solver:
        solver = TTestPower()

    def _get_sample_size_col(col):
        
        sd = df[col].std()
        mean = df[col].mean()
        es = (rel_effect_size*mean)/sd

        return solver.solve_power(effect_size=es, alpha=alpha, power=power, nobs=None)
    metric_names = [m.name for m in metrics_list]
    x = {col : _get_sample_size_col(col) for col in metric_names}
    return {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}