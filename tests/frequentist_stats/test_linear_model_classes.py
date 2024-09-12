from string import ascii_letters
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import mozanalysis.frequentist_stats.linear_models as mafslm

from .test_linear_model_functions import test_model, test_model_covariate


def test_from_formula():
    # no covariate

    actual = mafslm.MozOLS.from_formula(test_model.formula, test_model.model_df)

    # test that constructed design matrices are identical
    assert np.isclose(actual.exog, test_model.results.model.exog).all()
    assert np.isclose(actual.endog, test_model.results.model.endog).all()

    assert actual.formula == test_model.formula

    pd.testing.assert_frame_equal(actual.data.frame, test_model.model_df)

    assert actual.k_constant == test_model.results.model.k_constant

    # with covariate

    actual = mafslm.MozOLS.from_formula(
        test_model_covariate.formula, test_model_covariate.model_df
    )

    assert np.isclose(actual.exog, test_model_covariate.results.model.exog).all()
    assert np.isclose(actual.endog, test_model_covariate.results.model.endog).all()

    assert actual.formula == test_model_covariate.formula

    pd.testing.assert_frame_equal(actual.data.frame, test_model_covariate.model_df)

    assert actual.k_constant == test_model_covariate.results.model.k_constant


def _compare_models(actual, expected):
    assert np.isclose(actual.params, expected.params).all()
    assert np.isclose(actual.cov_params(), expected.cov_params()).all()
    assert np.isclose(
        actual.normalized_cov_params, expected.normalized_cov_params
    ).all()
    assert np.isclose(actual.bse, expected.bse).all()

    assert np.isclose(actual.llf, expected.llf)


def test_fit_basic(monkeypatch):
    # since sm.OLS inherts a fit method from statsmodels.base.LikelihoodModel
    # these checks ensure that MozOLS calls our fit method, not the original
    pinv_mock = MagicMock()
    monkeypatch.setattr("numpy.linalg.pinv", pinv_mock)
    # no covariate
    model = mafslm.MozOLS.from_formula(test_model.formula, test_model.model_df)
    actual = model.fit()
    pinv_mock.assert_not_called()

    _compare_models(actual, test_model.results)

    # with covariate
    model = mafslm.MozOLS.from_formula(
        test_model_covariate.formula, test_model_covariate.model_df
    )
    actual = model.fit()
    pinv_mock.assert_not_called()

    _compare_models(actual, test_model_covariate.results)


def test_fit_random():
    np.random.seed = 42
    for n_col in range(2, 10):
        for n_row in range(100, 10_000, 1000):
            model_df = pd.DataFrame(
                np.random.random(size=(n_row, n_col)),
                columns=list(ascii_letters[:n_col]),
            )
            formula = "a ~ "
            for letter in ascii_letters[1:n_col]:
                formula += f"{letter} + "
            formula = formula[:-3]

            expected = smf.ols(formula, model_df).fit()
            actual = mafslm.MozOLS.from_formula(formula, model_df).fit()

            _compare_models(actual, expected)
