"""Run our Algorithm 8.2 on the Boston data and compare with sklearn (ISLP lab 8.3.4).

Usage:  python boston_demo.py
Writes test_error_vs_trees.png next to this script.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ISLP import load_data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from boosting import BoostedRegressionTrees

# Same data and train/test split as the ISLP lab (Section 8.3.2).
Boston = load_data("Boston")
X = Boston.drop(columns="medv").to_numpy()
y = Boston["medv"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


def test_mse_path(model):
    return np.array([np.mean((y_test - pred) ** 2)
                     for pred in model.staged_predict(X_test)])


B, LAM = 5000, 0.01

# 1. Our implementation, for a few interaction depths d.
paths = {}
for d in (1, 2, 4):
    model = BoostedRegressionTrees(n_trees=B, learning_rate=LAM, n_splits=d,
                                   random_state=0).fit(X_train, y_train)
    paths[f"ours, d={d}"] = test_mse_path(model)

# 2. sklearn with matching settings (d=4 splits -> 5 leaves). The main difference
#    left is that sklearn starts f(x) at mean(y) instead of 0.
sk = GradientBoostingRegressor(n_estimators=B, learning_rate=LAM,
                               max_leaf_nodes=4 + 1, max_depth=None,
                               random_state=0)
sk.fit(X_train, y_train)
paths["sklearn GBR, d=4"] = test_mse_path(sk)

print(f"Test MSE after B={B} trees, lambda={LAM}")
for name, path in paths.items():
    print(f"  {name:18s} final {path[-1]:6.2f}   best {path.min():6.2f} "
          f"(at B={path.argmin() + 1})")

fig, ax = plt.subplots(figsize=(8, 5))
for name, path in paths.items():
    ax.plot(np.arange(1, B + 1), path, label=name,
            linestyle="--" if name.startswith("sklearn") else "-")
ax.set_xlabel("Number of trees B")
ax.set_ylabel("Test MSE")
ax.set_ylim(0, 60)
ax.set_title(f"Boosting on Boston (lambda = {LAM})")
ax.legend()
out = Path(__file__).with_name("test_error_vs_trees.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved plot to {out}")
