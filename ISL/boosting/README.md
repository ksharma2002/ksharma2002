# Boosting (ISLP Section 8.2.3)

A from-scratch version of **Algorithm 8.2, Boosting for Regression Trees**, from
*An Introduction to Statistical Learning with Applications in Python*, compared
with sklearn's `GradientBoostingRegressor` on the Boston data (lab 8.3.4).

| File | What it is |
|------|------------|
| `boosting.py` | `BoostedRegressionTrees`: the residual-fitting loop of Algorithm 8.2 |
| `boston_demo.py` | Fits it on Boston for d = 1, 2, 4 and compares with sklearn |
| `test_error_vs_trees.png` | Test MSE vs. number of trees, produced by the demo |

## Run it

```bash
pip install scikit-learn pandas matplotlib
pip install --no-deps ISLP   # only needed for load_data("Boston")
python boston_demo.py
```

## Notes

- **d splits = d + 1 leaves.** The book's d maps to `max_leaf_nodes=d+1` in sklearn, not `max_depth=d`.
- **Starting point.** The book starts at f(x) = 0; sklearn starts at mean(y). That only changes the first few hundred trees.
- **Default λ.** The lab text says sklearn's default learning rate is 0.001, but the actual default `learning_rate` is 0.1.

## Results (B = 5000, λ = 0.01, same 70/30 split as the lab)

| Model | Final test MSE | Best test MSE (at B) |
|-------|---------------:|---------------------:|
| ours, d = 1 | 19.69 | 18.67 (1869) |
| ours, d = 2 | 17.10 | 17.09 (4702) |
| ours, d = 4 | 14.05 | 13.59 (1950) |
| sklearn GBR, d = 4 | 14.03 | 13.57 (1950) |

For reference, the book's lab gets 14.48 with `max_depth=3` (up to 7 splits) and λ = 0.001.
