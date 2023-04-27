# lasso
Self-implemented version of LASSO

## Comments
- Accelerated by 
  - pre-storage
  - normalization
- Resulted in only an order of magnitude slower than LASSO in sklearn

## Example
```Python
from lasso import Lasso

# general fit
lmd = 0.1
model = Lasso()
model.fit(y, X, lmd)
model.predict(X)
model.score(y, X)

# draw beta convergence path
model = Lasso()
model.path_fit(y, X)
model.draw_beta_path()
```