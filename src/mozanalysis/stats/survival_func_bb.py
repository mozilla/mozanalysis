import numpy as np
import mozanalysis.stats.bayesian_binary as bb


def get_thresholds(col, max_num_thresholds=101):
    """Return a set of interesting thresholds for the dataset `col`

    Assumes that the values are non-negative, with zero as a special case.

    Args:
        col: a Series of individual data for a metric
        max_num_thresholds (int): Return at most this many threshold values.

    Returns:
        A list of thresholds. By default these are de-duped percentiles
        of the nonzero data.
    """
    # When taking quantiles, treat "0" as a special case so that we
    # still have resolution if 99% of users are 0.
    nonzero_quantiles = col[col > 0].quantile(
        np.linspace(0, 1, max_num_thresholds)
    )
    return sorted(
        [np.float64(0)] + list(nonzero_quantiles.unique())
    )[:-1]  # The thresholds get used as `>` not `>=`, so exclude the max value


def crunch_nums_survival(df, col_label, ref_branch_label='control', thresholds=None):
    if not thresholds:
        thresholds = get_thresholds(df[col_label])

    res = {
        'comparative': {
            b: {
                x: None for x in thresholds
            } for b in df.branch.unique() if b != ref_branch_label
        },
        'individual': {
            b: {
                x: None for x in thresholds
            } for b in df.branch.unique()
        }
    }

    for x in thresholds:
        assert 'tmp_crunch_nums' not in df.columns
        try:
            # Sorry for mutating the input inplace
            df['tmp_crunch_nums'] = df[col_label] > x
            bla = bb.compare(
                df, 'tmp_crunch_nums', ref_branch_label=ref_branch_label
            )
        finally:
            df.drop('tmp_crunch_nums', axis='columns', inplace=True)

        for branch, data in bla['comparative'].items():
            res['comparative'][branch][x] = data
        for branch, data in bla['individual'].items():
            res['individual'][branch][x] = data

    return res


# TODO: move this to utils?
def all_eq(iterable):
    return len(set(iterable)) <= 1
