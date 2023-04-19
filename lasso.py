import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


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
        self.l = None  # attempts number with different lambda's
        self.y = None  # target, n*1
        self.X = None  # feature, n*p
        self.features = None  # feature names, 1*j
        self.beta = None  # array of beta_j, 1*j
        self.beta_0 = None  # beta_0, scalar
        self.lmd = None  # lambda
        self.dof = None  # degree of freedom
        self.max_iter = max_iter  # maximum iterations
        self.stop_crit = stop_crit  # stopping criterion
        self.shifts = None  # centralization changes,
        self.scale = None  # scaling change, scalar
        self.y_bar = None  # sample mean of y, scalar
        self.X_bar = None  # sample means of X_j, 1*j
        self.y_X = None  # inner products of y and X_j, 1*j
        self.X_X = None  # inner products of X_k and X_j, j*j
        self.lmd_min = None  # minimum lambda
        self.lmd_max = None  # maximum lambda
        self.step_size = None  # step size of lambda change
        self.lmd_path = None  # path of lambda, 1*l
        self.beta_path = None  # pathes of beta, p*l
        self.beta_0_path = None  # path of beta_0, 1*l
        self.resid_path = None  # pathes of in-sample residual of samples, n*l
        self.se_path = None  # pathes of in-sample squared error of samples, n*l
        self.mse_path = None  # path of in-sample mean squared error, 1*l

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
        b: int | float | np.ndarray, gamma: int | float | np.ndarray
    ) -> float | np.ndarray:
        """Soft-threshold function, S(b, gamma) = sign(b) * max(0, |b| - gamma), which
        is the minimizer of (1/2) * (x-b)^2 + gamma * |x|

        Parameters
        ----------
        b : int | float | np.ndarray
            `b` in S(b, gamma)
        gamma : int | float | np.ndarray
            `gamma` in S(b, gamma)

        Returns
        -------
        float | np.ndarray
            S(b, gamma)
        """
        if isinstance(gamma, np.ndarray) and not isinstance(b, np.ndarray):
            b = np.full(gamma.shape, b)
        elif isinstance(b, np.ndarray) and not isinstance(gamma, np.ndarray):
            gamma = np.full(b.shape, gamma)
        base = np.zeros(b.shape) if isinstance(b, np.ndarray) else 0
        return np.sign(b) * np.maximum(base, np.abs(b) - gamma)

    def _build(self) -> None:
        """Build the model

        Normalization (to speed up the convergence of `beta` and `beta_0`)
            1) centralize: minus each `X_j` by different numbers, `self.shifts`
            2) scale: divide all `X_j` and `y` by the same number, `self.scale`

        Precomputation (to speed the iteration process in coordinate descent)
            1) sample means: `self.y_bar` and `self.X_bar`
            2) inner products: `self.y_X`, `self.o_X` (all 0), and `self.X_X`

        Initialization of `beta_0` and `beta`
            1) `beta_0` = `y_bar` is already the final solution given `X_j` = 0
            2) initialize each `beta_j` in `beta` randomly
        """
        # normalize
        self.shifts = self.X.mean(axis=0)
        self.X -= self.shifts.reshape((1, self.p))
        self.scale = self.X.std(axis=0).mean()
        self.X /= self.scale
        self.y /= self.scale
        self.lmd /= self.scale**2
        # precompute
        self.y_bar, self.X_bar = self.y.mean(), np.zeros(self.p)
        self.y_X, self.X_X = self.y.dot(self.X), self.X.T @ self.X
        # initialize beta_0 and beta
        self.beta_0 = self.y_bar  # final solution
        self.beta = np.random.randn(self.p)  # randomly initialized

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
        res = self.y_X[j] + self.beta[j] * self.X_X[j, j]
        res -= (self.beta.reshape((self.p, 1)) * self.X.T).dot(self.X[:, j]).sum()
        return res

    def _update_beta(self) -> None:
        """Update `self.beta`
        
        minimize (1/2)*(beta_j - \hat{beta_j})^2 + gamma * |beta_j|
        => beta_j^star = S(\hat{beta_j}, gamma)
        where
            \hat{beta_j} = <Z_j, X_j> / ||X_j||^2
            Z_j = (Z_1j, ..., Z_nj)^T
            gamma = n * lamda / ||X_j||^2
        """
        for j in range(self.p):
            Z_j_X_j = self._inner_prod_z_x(j)
            X_j_X_j = self.X_X[j, j]
            beta_j_hat = Z_j_X_j / X_j_X_j
            gamma = self.n * self.lmd / X_j_X_j
            self.beta[j] = Lasso.soft_thresh(beta_j_hat, gamma)

    def _run(self) -> None:
        """Run the model
            1) coordinate descent
            2) adjustment
            3) calculate degree of freedom
        """
        # coordinate descent
        for _ in range(self.max_iter):
            all_beta_old = np.insert(self.beta, 0, self.beta_0)
            self._update_beta()
            all_beta_new = np.insert(self.beta, 0, self.beta_0)
            chg = all_beta_new - all_beta_old
            abs_pct_chg = np.abs(Lasso.protected_div(chg, all_beta_old))
            if (abs_pct_chg < self.stop_crit).all():
                break
        # adjust beta_0 and lambda affected by normalization, keep y and X the same
        self.beta_0 = self.beta_0 * self.scale - self.shifts.dot(self.beta)
        self.lmd *= self.scale**2
        # degree of freedom
        self.dof = np.sum(np.abs(self.beta))
        
    def fit(
        self,
        y: np.ndarray | pd.DataFrame,
        X: np.ndarray | pd.DataFrame,
        lmd: int | float,
    ) -> "Lasso":
        """Fit the model

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
        self.features = X.columns if isinstance(X, pd.DataFrame) else None
        self.y, self.X = np.array(y), np.array(X)
        self.n, self.p = X.shape
        self.lmd = lmd
        # build and run
        self._build()
        self._run()
        return self

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

    def score(
        self,
        y: np.ndarray | pd.DataFrame,
        X: np.ndarray | pd.DataFrame,
    ) -> float:
        if self.beta is None:
            raise Exception("model has not been fitted")
        return mean_squared_error(y, self.predict(X))

    def refit(
        self,
        lmd: int | float,
    ) -> "Lasso":
        """Refit the model with a new `lambda` but the original `y` and `X`

        Parameters
        ----------
        lmd : int | float
            `lambda` as the regularization parameter

        Returns
        -------
        Lasso
            The model itself
        """
        # reset lambda
        self.lmd = lmd / self.scale**2
        # reset beta and beta_0
        self.beta_0 = self.y_bar
        self.beta = np.random.randn(self.p)
        # run
        self._run()
        return self

    def path_fit(
        self,
        y: np.ndarray | pd.DataFrame,
        X: np.ndarray | pd.DataFrame,
        lmd_min: int | float | None = None,
        lmd_max: int | float | None = None,
        step_size: int | float = 0.1,
    ) -> "Lasso":
        """Fit the model with different `lambda`'s to get pathes of `beta`

        Parameters
        ----------
        y : np.ndarray | pd.DataFrame
            Targets
        X : np.ndarray | pd.DataFrame
            Features
        lmd_min : int | float | None, optional
            Minimum `lambda` (including), by default None
        lmd_max : int | float | None, optional
            Maximum `lambda` (not including), by default None
        step_size : int | float, optional
            Step size of `lambda` change, by default 0.1

        Returns
        -------
        Lasso
            The model itself
        """
        if lmd_min is None:
            lmd_min = 0
        can_stop = True
        if lmd_max is None:
            can_stop = False
            lmd_max = float("inf")
        if lmd_min < 0:
            raise Exception("`lmd` should not be less than 0")
        if lmd_max < lmd_min:
            raise Exception("`lmd_max` should not be less than `lmd_min`")
        # assign attributes
        self.features = X.columns if isinstance(X, pd.DataFrame) else None
        self.y, self.X = np.array(y), np.array(X)
        self.n, self.p = X.shape
        self.step_size = step_size
        self.lmd = lmd = lmd_min
        # build
        self._build()
        # iterate lambda
        lmd_path, dof_path = [], []
        beta_path, beta_0_path, resid_path = [], [], []
        flag = True
        while (flag or can_stop) and lmd < lmd_max:
            self.refit(lmd)
            lmd_path.append(self.lmd)
            dof_path.append(self.dof)
            beta_path.append(self.beta)
            beta_0_path.append(self.beta_0)
            resid_path.append(y - self.predict(X))
            lmd += step_size
            flag = (np.abs(beta_path[-1]) > 1e-9).any()
        # store results
        self.l = len(lmd_path)
        self.lmd_min, self.lmd_max = lmd_min, lmd
        self.lmd_path = np.array(lmd_path)
        self.dof_path = np.array(dof_path)
        self.beta_path = np.array(beta_path).T
        self.beta_0_path = np.array(beta_0_path)
        self.resid_path = np.array(resid_path).T
        self.se_path = self.resid_path**2
        self.mse_path = self.se_path.mean(axis=0)
        return self

    def draw_beta_path(self) -> None:
        """Draw the pathes of `beta` with respect to different `lambda`'s"""
        if self.beta_path is None:
            raise Exception("`path_fit()` has not been called")
        # set legends
        lgds = list(range(1, 1 + self.p)) if self.features is None else self.features
        # plot
        plt.figure(figsize=(12, 8))
        for j in range(self.p):
            plt.plot(self.lmd_path, self.beta_path[j], label=lgds[j])
        plt.xlabel("lambda")
        plt.ylabel("beta")
        plt.legend(
            bbox_to_anchor=(0, 1, 1, 0),
            loc="lower left",
            mode="expand",
            ncol=5,
        )
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import time

    n, p = 100, 20
    X = np.random.uniform(10, 20, (n, p))
    eps = np.random.randn(n)
    beta_true = np.random.randint(-10, 10, p)
    beta_0_true = 0
    y = beta_0_true + X.dot(beta_true) + eps

    # from sklearn.linear_model import Lasso as L
    # m = L(alpha=1)
    # start = time.time()
    # m.fit(X, y)
    # print(m.intercept_)
    # print(m.coef_)
    # print(time.time() - start, "t")

    # lasso = Lasso()
    # start = time.time()
    # lasso.fit(y, X, 2)
    # lasso.refit(1)
    # print(lasso.beta_0)
    # print(lasso.beta)
    # print(time.time() - start, "t")

    lasso = Lasso()
    start = time.time()
    lasso.path_fit(y, X, step_size=0.2)
    print(time.time() - start)
    lasso.draw_beta_path()
