"""
Mock implementation of JumpModel for demonstration purposes
This simulates the behavior of a Jump Model for the CJM project
"""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import pandas as pd


class JumpModel:
    """
    Mock Jump Model implementation using Gaussian Mixture Model as base
    with jump penalty incorporated through transition constraints
    """

    def __init__(self, n_components=2, jump_penalty=1.0, cont=False,
                 grid_size=0.01, mode_loss=False, n_init=10,
                 max_iter=1000, tol=1e-8, random_state=None):
        self.n_components = n_components
        self.jump_penalty = jump_penalty
        self.cont = cont
        self.grid_size = grid_size
        self.mode_loss = mode_loss
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Internal models
        self._gmm = GaussianMixture(
            n_components=n_components,
            n_init=n_init,
            max_iter=max_iter,
            tol=float(tol),  # Ensure tol is float
            random_state=random_state
        )

    def fit(self, X, ret_ser=None, sort_by="cumret"):
        """
        Fit the Jump Model

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        ret_ser : array-like, optional
            Return series for sorting
        sort_by : str
            Sorting method ('cumret' or other)
        """
        # Fit base GMM
        self._gmm.fit(X)

        # Calculate cluster centers
        self.centers_ = self._gmm.means_

        # Simulate transition matrix with jump penalty effect
        # Higher penalty = fewer transitions
        persistence = 1.0 / (1.0 + np.exp(-np.log10(self.jump_penalty)))
        off_diag = (1.0 - persistence) / (self.n_components - 1)

        self.transmat_ = np.full((self.n_components, self.n_components), off_diag)
        np.fill_diagonal(self.transmat_, persistence)

        # Sort components by cumulative return if requested
        if sort_by == "cumret" and ret_ser is not None:
            labels = self._gmm.predict(X)
            cum_rets = []
            for k in range(self.n_components):
                mask = (labels == k)
                if mask.sum() > 0:
                    cum_ret = ret_ser[mask].sum() if hasattr(ret_ser, '__getitem__') else 0
                else:
                    cum_ret = 0
                cum_rets.append(cum_ret)

            # Sort components by cumulative return
            sort_idx = np.argsort(cum_rets)
            self.centers_ = self.centers_[sort_idx]

            # Reorder GMM components
            self._gmm.means_ = self._gmm.means_[sort_idx]
            self._gmm.covariances_ = self._gmm.covariances_[sort_idx]
            self._gmm.weights_ = self._gmm.weights_[sort_idx]

        return self

    def predict(self, X):
        """
        Predict labels for discrete Jump Model

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix

        Returns:
        --------
        labels : array of shape (n_samples,)
            Predicted cluster labels
        """
        if not self.cont:
            # Discrete model - return hard labels
            return self._gmm.predict(X)
        else:
            # For continuous model, still return argmax of probabilities
            proba = self.predict_proba(X)
            return proba.values.argmax(axis=1)

    def predict_proba(self, X):
        """
        Predict probabilities for continuous Jump Model

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix

        Returns:
        --------
        proba : DataFrame
            Predicted probabilities for each component
        """
        proba = self._gmm.predict_proba(X)

        if self.cont and self.jump_penalty > 1.0:
            # Apply jump penalty smoothing for continuous model
            # Higher penalty = smoother transitions
            n_samples = X.shape[0]
            smoothed_proba = np.zeros_like(proba)

            # Apply temporal smoothing based on jump penalty
            alpha = 1.0 / (1.0 + self.jump_penalty / 10.0)
            smoothed_proba[0] = proba[0]

            for t in range(1, n_samples):
                smoothed_proba[t] = alpha * proba[t] + (1 - alpha) * smoothed_proba[t-1]

            # Renormalize
            smoothed_proba = smoothed_proba / smoothed_proba.sum(axis=1, keepdims=True)
            proba = smoothed_proba

        # Return as DataFrame for compatibility
        return pd.DataFrame(proba, columns=[f'prob_{i}' for i in range(self.n_components)])