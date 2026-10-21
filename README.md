# custom_loss

Custom LightGBM objectives written as a plain JAX loss function. JAX autodiff
supplies the gradient and the diagonal Hessian that LightGBM needs, so a new
loss is one method, not a hand-derived gradient.

The main use is distributional regression: predict a full multivariate
Gaussian (means, variances and correlations) for several targets at once.

## Files

| File | What it is |
|---|---|
| `lgb_custom_loss_template.py` | `LGBCustomLoss` base class. Subclass it and implement `loss_fn(y_true, y_pred)`. Includes MSE, Gaussian NLL and multivariate Gaussian NLL examples. |
| `lgb_multivariategaussian_loss.py` | Standalone multivariate Gaussian loss (the first version of the idea). |
| `mvgmodel.py` | `MVGModel`, a fit/predict wrapper that returns a mean vector and a covariance matrix per row. It also takes an optional weight matrix, to score linear combinations of the targets. |

## Quick start

```bash
pip install jax lightgbm scikit-learn numpy
python lgb_custom_loss_template.py
```

Write your own loss:

```python
import jax.numpy as jnp
from lgb_custom_loss_template import LGBCustomLoss

class HuberLoss(LGBCustomLoss):
    def __init__(self, num_labels=1, delta=1.0):
        super().__init__(num_labels=num_labels, num_preds=num_labels,
                         loss_name="Huber")
        self.delta = delta

    def loss_fn(self, y_true, y_pred):
        r = jnp.abs(y_true - y_pred)
        return jnp.sum(jnp.where(r < self.delta, 0.5 * r ** 2,
                                 self.delta * (r - 0.5 * self.delta)))
```

Then train it the way the `__main__` block of `lgb_custom_loss_template.py`
does: stack the data with `stack_x` / `stack_y` and pass `loss.obj` as the
objective and `loss.loss` as the eval metric.

## How it works

LightGBM grows one tree per output. The template tiles the feature rows so
that each output head gets its own copy, and reshapes LightGBM's flat
column-major predictions back to `(N, num_preds)`. `jax.grad` gives the
gradient of the per-sample loss and `jax.jacobian` of that gradient gives the
Hessian, of which only the diagonal is passed to LightGBM.

## Limitations

- Only the diagonal of the Hessian is used. This is what LightGBM accepts, but
  it ignores cross-output curvature.
- The multivariate Gaussian builds correlations with `tanh`. That keeps each
  correlation in (-1, 1) but does not guarantee a positive-definite matrix for
  3 or more targets, so the loss can become NaN on long or overfitting runs.
  A Cholesky parameterisation would fix this.
- The Hessian is a full Jacobian per row, so cost grows with the square of the
  number of outputs.

## License

MIT
