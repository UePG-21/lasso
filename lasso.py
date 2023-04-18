import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Lasso:
    """LASSO regression

    Model:
        y_i = beta_0 + beta.T @ X_i + epsilon_i, or
        y_i = beta_0 + sum_{j=1}^{p}{X_ij * beta_j} + epsilon_i
    where
        i = 1, ..., n as sample subscript
        j = 1, ..., p as feature subscript

    Optimization object:
        L(beta_0, beta) = (1/2n) * ||y - beta_0 - X @ beta||_2^2 + lambda * ||beta||_1
    where
        ||·||_1 means L1-norm
        ||·||_2 means L2-norm

    Optimization method: coordinate descent
    """

    def __init__(self, max_iter: int = 1000, stop_crit: float = 0.001) -> None:
        """Initialization

        Parameters
        ----------
        max_iter : int, optional
            The maximum number of iterations, by default 1000
        stop_crit : float, optional
            Stopping criterion for beta_0 and beta update, which is if all bete's
            absolute update percentages are less than this number in a coordinate
            descent loop, by default 0.001
        """
        self.n = None  # sample size
        self.p = None  # feature size
        self.y = None  # target, n*1
        self.X = None  # feature, n*p
        self.features = None  # feature names
        self.beta = None  # array of beta_j
        self.beta_0 = None  # beta_0
        self.lmd = None  # lambda
        self.max_iter = max_iter  # maximum iterations
        self.stop_crit = stop_crit  # stopping criterion
        self.y_bar = None  # sample mean of y
        self.X_bar = None  # sample means of X_j
        self.y_X = None  # inner products of y and X_j
        self.o_X = None  # inner products of 1 and X_j
        self.X_X = None  # inner products of X_k and X_j
        self.lmd_min = None  # minimum lambda
        self.lmd_max = None  # maximum lambda
        self.lmd_path = None  # path of lambda, 1*l
        self.beta_path = None  # pathes of beta, p*l
        self.beta_0_path = None  # pathes of beta_0, 1*l

    @staticmethod
    def protected_div(
        num: float | np.ndarray, den: float | np.ndarray
    ) -> float | np.ndarray:
        """Calculate protected division, avoiding denominator being 0

        Parameters
        ----------
        num : float | np.ndarray
            Numerator
        den : float | np.ndarray
            Denominator

        Returns
        -------
        float | np.ndarray
            Division
        """
        den = np.where(np.abs(den) > 1e-6, den, 1)
        return num / den

    @staticmethod
    def soft_thresh(
        b: float | np.ndarray, gamma: float | np.ndarray
    ) -> float | np.ndarray:
        """Soft-threshold function, S(b, gamma) = sign(b) * max(0, |n| - gamma), which
            is the minimizer of (1/2) * (x-b)^2 + gamma * |x|

        Parameters
        ----------
        b : float | np.ndarray
            `b` in S(b, gamma)
        gamma : float | np.ndarray
            `gamma` in S(b, gamma)

        Returns
        -------
        float | np.ndarray
            S(b, gamma)
        """
        if type(b) is not type(gamma):
            raise Exception("type of `b` should be the same as type of `gamma`")
        base = np.zeros((len(b),)) if isinstance(b, np.ndarray) else 0.0
        return np.sign(b) * np.maximum(base, np.abs(b) - gamma)

    def _sample_means(self) -> tuple[np.ndarray]:
        """Calculate sample means `self.y_bar` and `self.X_bar`

        Returns
        -------
        tuple[np.ndarray]
            `self.y_bar` and `self.X_bar`
        """
        return self.y.mean(), self.X.mean(axis=0)

    def _inner_prods(self) -> tuple[np.ndarray]:
        """Calculate inner products `self.y_X`, `self.o_X`, and `self.X_X`

        Returns
        -------
        tuple[np.ndarray]
            `self.y_X`, `self.o_X`, and `self.X_X`
        """
        return self.y.dot(self.X), np.ones(self.n).dot(self.X), self.X.T @ self.X

    def _inner_prod_z_x(self, j: int) -> float:
        """Calculate `<Z_j, X_j>` as the inner product of `Z_j` and `X_j`

        Parameters
        ----------
        j : int
            Feature index

        Returns
        -------
        float
            Inner product of `Z_j` and `X_j`
        """
        res = self.y_X[j] - self.beta_0 * self.o_X[j]
        res -= (self.beta.reshape((self.p, 1)) * self.X.T).dot(self.X[:, j]).sum()
        res += self.beta[j] * self.X_X[j, j]
        return res

    def _initalize_beta(self) -> None:
        """Initialize `self.beta_0` and `self.beta`"""
        self.beta_0 = np.random.randn()
        self.beta = np.random.randn(self.p)

    def _update_beta_j(self, j: int) -> None:
        """Update `beta_j` in `self.beta`

        minimize (1/2)*(beta_j - \hat{beta_j})^2 + gamma * |beta_j|
        => beta_j^star = S(\hat{beta_j}, gamma)
        where
            \hat{beta_j} = <Z_j, X_j> / ||X_j||^2
            Z_j = (Z_1j, ..., Z_nj)^T
            gamma = n * lamda / ||X_j||^2

        Parameters
        ----------
        j : int
            Feature index
        """
        Z_j_X_j = self._inner_prod_z_x(j)
        X_j_X_j = self.X_X[j, j]
        beta_j_hat = Z_j_X_j / X_j_X_j
        gamma = self.n * self.lmd / X_j_X_j
        self.beta[j] = Lasso.soft_thresh(beta_j_hat, gamma)

    def _update_beta(self) -> None:
        """Update `self.beta`"""
        for j in range(self.p):
            self._update_beta_j(j)

    def _update_beta_0(self) -> None:
        """Update `self.beta_0`"""
        self.beta_0 = self.y_bar - self.beta.dot(self.X_bar)

    def _coordinate_descent(self) -> None:
        """Coordinate descent"""
        for _ in range(self.max_iter):
            all_beta_old = np.insert(self.beta, 0, self.beta_0)
            self._update_beta()
            self._update_beta_0()
            all_beta_new = np.insert(self.beta, 0, self.beta_0)
            chg = all_beta_new - all_beta_old
            abs_pct_chg = np.abs(Lasso.protected_div(chg, all_beta_old))
            if (abs_pct_chg < self.stop_crit).all():
                break

    def fit(
        self,
        y: np.ndarray | pd.DataFrame,
        X: np.ndarray | pd.DataFrame,
        lmd: int | float,
    ) -> "Lasso":
        """Fit model

        Parameters
        ----------
        y : np.ndarray | pd.DataFrame
            Targets
        X : np.ndarray | pd.DataFrame
            Features
        lmd : int | float
            lambda as the regularization parameter

        Returns
        -------
        Lasso
            The model itself
        """
        if y.ndim != 1:
            raise Exception("dimension of `y` should be 1")
        if y.shape[0] != X.shape[0]:
            raise Exception("length of `y` and `X` should be the same")
        # assign attributes
        self.y, self.X = np.array(y), np.array(X)
        self.features = X.columns if isinstance(X, pd.DataFrame) else None
        self.n, self.p = X.shape
        self.lmd = lmd
        # precompute
        self.y_bar, self.X_bar = self._sample_means()
        self.y_X, self.o_X, self.X_X = self._inner_prods()
        # initialize beta_0 and beta
        self._initalize_beta()
        # coordinate descent
        self._coordinate_descent()
        return self

    def refit(
        self,
        lmd: int | float,
    ) -> "Lasso":
        """Refit with a new `lambda`

        Parameters
        ----------
        lmd : int | float
            `lambda` as the regularization parameter

        Returns
        -------
        Lasso
            The model itself
        """
        # set new lambda
        self.lmd = lmd
        # initialize beta_0 and beta
        self._initalize_beta()
        # coordinate descent
        self._coordinate_descent()
        return self

    def path_fit(
        self,
        y: np.ndarray | pd.DataFrame,
        X: np.ndarray | pd.DataFrame,
        lmd_min: int | float | None = None,
        lmd_max: int | float | None = None,
        step_size: int | float = 0.1,
    ) -> "Lasso":
        """Fit model with different `lambda`'s to get pathes of `beta`

        Parameters
        ----------
        y : np.ndarray | pd.DataFrame
            Targets
        X : np.ndarray | pd.DataFrame
            Features
        lmd_min : int | float | None, optional
            Minimum `lambda`, by default None
        lmd_max : int | float | None, optional
            Maximum `lambda`, by default None
        step_size : int | float, optional
            Step size of `lambda` change, by default 0.1

        Returns
        -------
        Lasso
            The model itself
        """
        if lmd_min is None:
            lmd_min = 0
        if lmd_max is None:
            lmd_max = float("inf")
        if lmd_min < 0:
            raise Exception("`lmd` should not be less than 0")
        if lmd_max < lmd_min:
            raise Exception("`lmd_max` should not be less than `lmd_min`")
        # assign attributes
        self.y, self.X = np.array(y), np.array(X)
        self.features = X.columns if isinstance(X, pd.DataFrame) else None
        self.n, self.p = X.shape
        # precompute
        self.y_bar, self.X_bar = self._sample_means()
        self.y_X, self.o_X, self.X_X = self._inner_prods()
        # iterate lambda
        lmd_path, beta_path, beta_0_path = [], [], []
        lmd = lmd_min
        flag = True
        while flag and lmd <= lmd_max:
            self.refit(lmd)
            lmd_path.append(lmd)
            beta_path.append(self.beta)
            beta_0_path.append(self.beta_0)
            lmd += step_size
            flag = (np.abs(beta_path[-1]) > 1e-9).any()
        # store results
        self.lmd_min, self.lmd_max = lmd_min, lmd
        self.lmd_path = np.array(lmd_path)
        self.beta_path = np.array(beta_path).T
        self.beta_0_path = np.array(beta_0_path)
        return self

    def draw_beta_path(self, legend: bool = False) -> None:
        """Draw pathes of `beta` with respect to different `lambda`'s

        Parameters
        ----------
        legend : bool, optional
            Plot the legend or not, by default False
        """
        if self.beta_path is None:
            raise Exception("`path_fit()` has not been called")
        for j in range(self.p):
            beta_j_path = self.beta_path[j]
            plt.plot(self.lmd_path, beta_j_path)
        plt.xlabel("lambda")
        plt.ylabel("beta")
        if legend:
            if self.features is None:
                legends = list(range(1, 1 + self.p))
            else:
                legends = self.features
            plt.legend(legends, fontsize=8)
        plt.show()

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """Predict targets

        Parameters
        ----------
        X : np.ndarray | pd.DataFrame
            Features

        Returns
        -------
        np.ndarray | pd.DataFrame
            Predicted targets
        """
        if X.ndim != 2:
            raise Exception("dimension of `X` is invalid")
        if X.shape[1] != self.p:
            raise Exception("feature size of `X` does not match the training data")
        y_pred = self.beta_0 + X.dot(self.beta)
        return y_pred


if __name__ == "__main__":
    n, p = 1000, 3
    X = np.random.normal(0, 1, (n, p))
    eps = np.random.randn(n)
    beta_true = np.array([3, -17, 5])
    beta_0_true = 0
    y = beta_0_true + X.dot(beta_true) + eps

    lasso = Lasso()
    lasso.path_fit(y, X)
    lasso.draw_beta_path(True)
