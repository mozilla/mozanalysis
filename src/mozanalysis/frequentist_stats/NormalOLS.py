from typing import Literal

import numpy as np
from statsmodels.regression.linear_model import (
    OLS,
    OLSResults,
    RegressionResults,
    RegressionResultsWrapper,
)


class NormalOLS(OLS):
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
        y, X = self.wendog, self.wexog

        C = np.dot(X.T, X)

        # may throw np.linalg.LinAlgError if columns are collinear
        C_inv = np.linalg.inv(C)
        self.normalized_cov_params = C_inv

        L = np.linalg.cholesky(C)

        d = np.dot(X.T, y)

        sol = np.linalg.solve(L, d)
        beta = np.linalg.solve(L.T, sol)

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
