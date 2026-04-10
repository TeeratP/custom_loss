
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
from jax.config import config
config.update("jax_enable_x64", False)
import numpy as np
import lightgbm as lgb
import jax.numpy as jnp
from jax import vmap, jit, grad, jacobian, device_get
import jax


class LGBCustomLoss:
    """
    Template for a custom LightGBM loss using JAX autodiff.

    Subclass this and implement:
        - loss_fn(self, y_true, y_pred)  (per-sample scalar loss)
        - (optional) convert_pred(self, y_pred_flat) for inference-time output

    The base class handles:
        - Computing gradients and diagonal Hessians via JAX autodiff
        - Reshaping between LightGBM's flat format and (N, num_preds)
        - Providing obj() and loss() callbacks ready for lgb.LGBMModel
    """

    def __init__(self, num_labels: int, num_preds: int, loss_name: str = "custom_loss"):
        """
        Args:
            num_labels: number of target columns (y dimensions).
            num_preds:  number of raw outputs per sample that the model produces.
                        For a simple regression this equals num_labels.
                        For distributional models it can be larger (e.g. means + variances).
            loss_name:  string returned in the eval metric tuple.
        """
        self.num_labels = num_labels
        self.num_preds = num_preds
        self.loss_name = loss_name

    # ------------------------------------------------------------------
    # Override this
    # ------------------------------------------------------------------
    def loss_fn(self, y_true, y_pred):
        """
        Per-sample loss (scalar).

        Args:
            y_true: jnp array of shape (num_labels,)
            y_pred: jnp array of shape (num_preds,)

        Returns:
            scalar loss value (lower is better).
        """
        raise NotImplementedError("Subclasses must implement loss_fn")

    # ------------------------------------------------------------------
    # Override this (optional) for custom inference decoding
    # ------------------------------------------------------------------
    def convert_pred(self, y_pred_flat):
        """
        Convert the flat raw predictions coming out of LightGBM into
        whatever form is useful at inference time.

        Args:
            y_pred_flat: np array of shape (N * num_preds,) — LightGBM's raw
                         column-major output.

        Returns:
            User-defined; default implementation returns (N, num_preds) array.
        """
        return np.reshape(y_pred_flat, (-1, self.num_preds), order="F")

    # ------------------------------------------------------------------
    # Internals — usually no need to touch these
    # ------------------------------------------------------------------
    def _grad_hess(self, y_true, y_pred):
        grad_fn = jax.grad(self.loss_fn, argnums=1)
        hess_fn = jax.jacobian(grad_fn, argnums=1)
        hess_diag_fn = lambda y_true, y_pred: jnp.diag(hess_fn(y_true, y_pred))

        grads = vmap(grad_fn, in_axes=(0, 0))(y_true, y_pred)
        hess = vmap(hess_diag_fn, in_axes=(0, 0))(y_true, y_pred)
        return grads, hess

    def _unpack(self, y_true_flat, y_pred_flat):
        """Reshape LightGBM's flat arrays into (N, ...) matrices."""
        y_true = np.reshape(y_true_flat, (-1, self.num_preds))
        y_true = y_true[:, :self.num_labels]
        y_pred = np.reshape(y_pred_flat, (y_true.shape[0], self.num_preds))
        return y_true, y_pred

    def obj(self, y_true_flat, y_pred_flat):
        """
        Objective callback for lgb.LGBMModel(objective=...).

        Returns (grad, hess) both flattened to shape (N * num_preds,).
        """
        y_true, y_pred = self._unpack(y_true_flat, y_pred_flat)
        grads, hess = self._grad_hess(y_true, y_pred)

        grads = device_get(grads).flatten()
        hess = device_get(hess).flatten()
        return grads, hess

    def loss(self, y_true_flat, y_pred_flat):
        """
        Eval metric callback for lgb.LGBMModel.fit(eval_metric=...).

        Returns (name, value, is_higher_better).
        """
        y_true, y_pred = self._unpack(y_true_flat, y_pred_flat)

        is_higher_better = False
        vmap_loss = vmap(self.loss_fn, in_axes=(0, 0))
        value = float(jnp.mean(vmap_loss(y_true, y_pred)))

        return self.loss_name, value, is_higher_better


# ======================================================================
# Data helpers — stack features / labels for multi-output LightGBM
# ======================================================================
def stack_x(x, num_preds):
    """Tile feature matrix so each output head sees the same features."""
    x_stack = np.concatenate([x for _ in range(num_preds)], axis=1)
    return x_stack.reshape((-1, x.shape[1]))


def stack_y(y, num_labels, num_preds):
    """Tile label matrix to match the stacked feature layout."""
    num_dupe = int(num_preds / num_labels)
    y_stack = np.concatenate([y for _ in range(num_dupe)], axis=1)
    return y_stack.reshape((-1, 1))


# ======================================================================
# Example: Multivariate Gaussian negative log-likelihood
# ======================================================================
class MultiVariateGaussianLoss(LGBCustomLoss):
    def __init__(self, num_labels=5, y_train=None):
        num_corr = int(num_labels * (num_labels - 1) / 2)
        num_preds = num_labels + num_labels + num_corr

        super().__init__(
            num_labels=num_labels,
            num_preds=num_preds,
            loss_name="MVG_loss",
        )

        if y_train is not None:
            self.scale = np.std(y_train, axis=0)
            self.loc = np.mean(y_train, axis=0)
        else:
            self.scale = 1
            self.loc = 0

    def _get_distribution(self, y_pred):
        mu = y_pred[:self.num_labels] + self.loc
        var = jnp.exp(y_pred[self.num_labels:2 * self.num_labels]) * self.scale
        L = jnp.tanh(y_pred[2 * self.num_labels:])

        corr = jnp.zeros((self.num_labels, self.num_labels))
        idx = 0
        for row in range(self.num_labels):
            for col in range(row + 1, self.num_labels):
                corr = corr.at[row, col].add(L[idx])
                idx += 1

        corr = corr + corr.T + np.eye(self.num_labels)
        cov = var.reshape((self.num_labels, 1)) * var.reshape((1, self.num_labels)) * corr
        return mu, cov

    def loss_fn(self, y_true, y_pred):
        mu, cov = self._get_distribution(y_pred)
        llh = -0.5 * jnp.log(jnp.linalg.det(cov)) \
              - 0.5 * (y_true - mu).T @ jnp.linalg.inv(cov) @ (y_true - mu)
        return -llh

    def convert_pred(self, y_pred_flat):
        y_pred = np.reshape(y_pred_flat, (-1, self.num_preds), "F")
        dist_fn = vmap(self._get_distribution, in_axes=(0,))
        mu, cov = dist_fn(y_pred)
        return mu, cov


# ======================================================================
# Example: Simple MSE (for testing / as a minimal starting point)
# ======================================================================
class MSELoss(LGBCustomLoss):
    def __init__(self, num_labels=1):
        super().__init__(
            num_labels=num_labels,
            num_preds=num_labels,
            loss_name="MSE",
        )

    def loss_fn(self, y_true, y_pred):
        return jnp.mean((y_true - y_pred) ** 2)


# ======================================================================
# Example: Gaussian NLL (mean + variance per label)
# ======================================================================
class GaussianNLLLoss(LGBCustomLoss):
    """Each label gets a predicted mean and log-variance."""

    def __init__(self, num_labels=1):
        super().__init__(
            num_labels=num_labels,
            num_preds=num_labels * 2,
            loss_name="GaussianNLL",
        )

    def loss_fn(self, y_true, y_pred):
        mu = y_pred[:self.num_labels]
        log_var = y_pred[self.num_labels:]
        var = jnp.exp(log_var)
        return 0.5 * jnp.sum(log_var + (y_true - mu) ** 2 / var)

    def convert_pred(self, y_pred_flat):
        y_pred = np.reshape(y_pred_flat, (-1, self.num_preds), "F")
        mu = y_pred[:, :self.num_labels]
        var = np.exp(y_pred[:, self.num_labels:])
        return mu, var


# ======================================================================
# Usage example
# ======================================================================
if __name__ == "__main__":
    # --- Synthetic data for demonstration ---
    np.random.seed(42)
    N_train, N_test, n_features = 500, 100, 10
    num_labels = 3

    x_train = np.random.randn(N_train, n_features).astype(np.float32)
    y_train = np.random.randn(N_train, num_labels).astype(np.float32)
    x_test = np.random.randn(N_test, n_features).astype(np.float32)
    y_test = np.random.randn(N_test, num_labels).astype(np.float32)

    # --- Pick a loss (swap in your own subclass here) ---
    loss_obj = MSELoss(num_labels=num_labels)

    # --- Stack data for multi-output LightGBM ---
    x_train_s = stack_x(x_train, loss_obj.num_preds)
    y_train_s = stack_y(y_train, num_labels, loss_obj.num_preds)
    x_test_s = stack_x(x_test, loss_obj.num_preds)
    y_test_s = stack_y(y_test, num_labels, loss_obj.num_preds)

    # --- Train ---
    model = lgb.LGBMModel(
        objective=loss_obj.obj,
        num_iterations=200,
        learning_rate=0.05,
    )

    model.fit(
        x_train_s,
        y_train_s,
        eval_set=[(x_test_s, y_test_s)],
        eval_metric=loss_obj.loss,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(10)],
    )

    # --- Predict ---
    raw = model.predict(x_test_s)
    decoded = loss_obj.convert_pred(raw)
    print("Decoded predictions shape:", decoded.shape)
