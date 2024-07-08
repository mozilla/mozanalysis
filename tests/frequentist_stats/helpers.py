import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# all models in this file are fit using standard statsmodels.OLS objects
# so that they can be compared against MozOLS models


class UnitTestModel:
    def __init__(self):
        self.ref_branch = "treatment-a"
        self.alphas = [0.01, 0.05]
        self.branches = (
            ["control"] * 100 + ["treatment-a"] * 100 + ["treatment-b"] * 100
        )
        self.treatment_branches = ["control", "treatment-b"]
        searches = list(range(100)) + list(range(100, 200)) + list(range(200, 300))
        self.model_df = pd.DataFrame(
            {"search_count": searches, "branch": self.branches}
        )
        self.target = "search_count"
        self.formula = "search_count ~ C(branch, Treatment(reference='treatment-a'))"
        self.results = smf.ols(self.formula, self.model_df).fit()


class UnitTestModelCovariate:
    def __init__(self):
        np.random.seed(42)
        self.ref_branch = "treatment-a"
        self.alphas = [0.01, 0.05]
        self.treatment_branches = ["control", "treatment-b"]
        branches = ["control"] * 100 + ["treatment-a"] * 100 + ["treatment-b"] * 100
        y_base = np.random.normal(loc=2, scale=0.1, size=300)
        y_pre_adj = np.random.normal(loc=0, scale=0.02, size=300)
        te = np.concatenate(
            [
                np.random.normal(loc=0, scale=0.05, size=100),
                np.random.normal(loc=0.1, scale=0.05, size=100),
                np.random.normal(loc=0.2, scale=0.05, size=100),
            ]
        )
        self.model_df = pd.DataFrame(
            {
                "search_count": y_base + te,
                "branch": branches,
                "search_count_pre": y_base + y_pre_adj,
            }
        )
        self.formula = "search_count ~ C(branch, Treatment(reference='treatment-a')) + search_count_pre"  # noqa: E501
        self.target = "search_count"
        self.covariate = "search_count_pre"
        self.results = smf.ols(self.formula, self.model_df).fit()


test_model = UnitTestModel()
test_model_covariate = UnitTestModelCovariate()
