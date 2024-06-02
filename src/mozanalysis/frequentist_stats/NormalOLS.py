import statsmodels 
import logging
logger = logging.getLogger(__name__)

# # monkey patching
# def _convert_endog_exog(self, endog, exog=None):
#     logger.info("empty _convert_endog_exog")
#     return statsmodels.base.data.ModelData._convert_endog_exog(self, endog, exog)

# statsmodels.base.data.PandasData._convert_endog_exog = _convert_endog_exog

# def _get_xarr(self, exog):
#     logger.info("empty _get_xarr")    
#     return exog 

# statsmodels.base.data.ModelData._get_xarr = _get_xarr

# def _get_yarr(self, exog):
#     logger.info("empty _get_yarr")    
#     return exog 

# statsmodels.base.data.ModelData._get_yarr = _get_yarr

def _handle_constant(self, hasconst):
    self.k_constant = 1
    self.const_idx = 0

statsmodels.base.data.ModelData._handle_constant = _handle_constant

from typing import Literal

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import (
    OLS,
    OLSResults,
    RegressionResults,
    RegressionResultsWrapper,
)

from formulaic import Formula, ModelMatrices
from formulaic.materializers import PandasMaterializer



class NormalOLS(OLS):
    @classmethod
    def from_formula(
        cls, formula: str, data: pd.DataFrame, *args, **kwargs
    ) -> "NormalOLS":
        """
        Create a Model from a formula and dataframe. This uses Formulaic instead
        of Patsy to create sparse design matrices which can dramatically reduce
        memory requirements (especially in experiments with many branches).

        Parameters
        ----------
        formula : str
            The formula specifying the model.
        data : pd.DataFrame
            The data for the model.
        *args
            Additional positional argument that are passed to the model.
        **kwargs
            Additional positional argument that are passed to the model.
        """
        logger.info("creating model matrix")
        mm = PandasMaterializer(data).get_model_matrix(
            Formula(formula)
        )
        
        assert isinstance(
            mm, ModelMatrices
        ), "Expected to create multiple matrices, is the formula correct?"

        logger.info("creating y df")
        # y = pd.DataFrame.sparse.from_spmatrix(
        #     mm.lhs, columns=mm.lhs.model_spec.column_names
        # )

        logger.info("creating X df")
        # X = pd.DataFrame.sparse.from_spmatrix(
        #     mm.rhs, columns=mm.rhs.model_spec.column_names
        # )

        kwargs.update({"formula": formula, "design_info": mm.rhs.model_spec, "missing":"none", "hasconst":True})
                    
        logger.info("constructing model object")
        mod = cls(mm.lhs, mm.rhs, *args, **kwargs)
        mod.formula = formula
        mod.data.frame = data
        logger.info("returning model object")
        return mod

    def fit(
        self,
        cov_type: Literal[
            "nonrobust",
            "fixed scale",
            "HC0",
            "HC1",
            "HC2",
            "HC3",
            "HAC",
            "hac-panel",
            "hac-groupsum",
            "cluster",
        ] = "nonrobust",
        cov_kwds=None,
        use_t: bool | None = None,
    ) -> RegressionResults:
        """
        Fits the model by solving the normal equations. This is more memory efficient
        than the pinv or QR methods available in statsmodels for our particular design
        matrix structure: (n,m) with n >> m. In particular, n can be >= 1e7 while m is
        1 + (n branches - 1) + 1 (if using covariate). When running in prod with the
        built in fitting algorithms, Jetstream can run out of memory.

        See section 4, specifically algorithm 4.1 from:
        https://math.uchicago.edu/~may/REU2012/REUPapers/Lee.pdf

        See also: https://xkcd.com/1838/

        Parameters
        ----------
        use_t : bool, optional
            Flag indicating to use the Student's t distribution when computing
            p-values.  Default behavior depends on cov_type. See
            `linear_model.RegressionResults.get_robustcov_results` for
            implementation details.
        cov_type : str, optional
            See `regression.linear_model.RegressionResults` for a description
            of the available covariance estimators.
        cov_kwds : list or None, optional
            See `linear_model.RegressionResults.get_robustcov_results` for a
            description required keywords for alternative covariance
            estimators.

        Returns
        -------
        RegressionResults
            The model estimation results.
        """
        # columns = self.exog_names
        logger.info("NormalOLS.fit")
        y, X = self.wendog, self.wexog
        logger.info(f"y type: {type(y)}")
        logger.info(f"X type: {type(X)}")        
        logger.info("C")
        C = np.dot(X.T, X)
        #C = X.transpose().dot(X).todense()
        logger.info("inverting X")
        # may throw np.linalg.LinAlgError if columns are collinear
        C_inv = np.linalg.inv(C)
        self.normalized_cov_params = C_inv
        logger.info("cholesky")
        L = np.linalg.cholesky(C)
        logger.info("d")
        #d = X.transpose().dot(y).todense()
        d = np.dot(X.T, y)
        logger.info("solving")
        sol = np.linalg.solve(L, d)
        beta = np.linalg.solve(L.T, sol)

        if self.rank is None:
            self.rank = np.linalg.matrix_rank(C)
        if self._df_model is None:
            self._df_model = float(self.rank - self.k_constant)
        if self._df_resid is None:
            self.df_resid = self.nobs - self.rank

        lfit = OLSResults(
            self,
            beta,
            normalized_cov_params=self.normalized_cov_params,
            cov_type=cov_type,
            cov_kwds=cov_kwds,
            use_t=use_t,
        )

        return RegressionResultsWrapper(lfit)
