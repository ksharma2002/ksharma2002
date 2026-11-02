"""Boosting for regression trees, following Algorithm 8.2 in ISLP (p. 349).

    1. Set f(x) = 0 and r_i = y_i for all i in the training set.
    2. For b = 1, 2, ..., B, repeat:
       (a) Fit a tree f_b with d splits (d + 1 terminal nodes) to (X, r).
       (b) Update f:          f(x) <- f(x) + lambda * f_b(x)        (8.10)
       (c) Update residuals:  r_i  <- r_i  - lambda * f_b(x_i)      (8.11)
    3. Output the boosted model  f(x) = sum_b lambda * f_b(x)       (8.12)

The individual trees come from sklearn's DecisionTreeRegressor; everything
about *boosting* (the residual loop, shrinkage, summing) is written here.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor


class BoostedRegressionTrees:
    """Boosted regression trees, as in ISLP Algorithm 8.2.

    Parameters
    ----------
    n_trees : int
        B, the number of trees.
    learning_rate : float
        lambda, the shrinkage parameter.
    n_splits : int
        d, the number of splits in each tree (d = 1 gives stumps).
        A tree with d splits has d + 1 leaves, so this maps to sklearn's
        ``max_leaf_nodes=d + 1`` (not ``max_depth=d``).
    random_state : int or None
        Passed to each tree, only to break ties between equally good splits.
    """

    def __init__(self, n_trees=1000, learning_rate=0.01, n_splits=1,
                 random_state=None):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Step 1: f(x) = 0, so the residuals start out as y itself.
        residuals = y.copy()
        self.trees_ = []
        self.train_mse_ = np.empty(self.n_trees)

        # Step 2: fit B small trees in sequence, each one to the current residuals.
        for b in range(self.n_trees):
            tree = DecisionTreeRegressor(max_leaf_nodes=self.n_splits + 1,
                                         random_state=self.random_state)
            tree.fit(X, residuals)                                   # (a)
            self.trees_.append(tree)                                 # (b)
            residuals -= self.learning_rate * tree.predict(X)        # (c)
            # Residuals are y - f(x), so their mean square is the training MSE.
            self.train_mse_[b] = np.mean(residuals ** 2)
        return self

    def staged_predict(self, X):
        """Yield the model's predictions after 1, 2, ..., B trees."""
        X = np.asarray(X, dtype=float)
        f = np.zeros(X.shape[0])
        for tree in self.trees_:
            f += self.learning_rate * tree.predict(X)
            yield f.copy()

    def predict(self, X):
        """Step 3: f(x) = sum over b of lambda * f_b(x)."""
        X = np.asarray(X, dtype=float)
        return self.learning_rate * sum(tree.predict(X) for tree in self.trees_)
