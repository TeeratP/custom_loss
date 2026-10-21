
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import datetime
import numpy as np
import lightgbm as lgb
import jax.numpy as jnp
from jax import vmap, jit, grad, random, jacobian
import jax
jax.config.update("jax_enable_x64", False)
from jax import device_get, ops

# custom loss function for light gbm
class MultiVariateGaussianLoss:
    def __init__(self, num_labels = 5, y_train = None):
        self.num_labels = num_labels
        num_mean = num_labels
        num_var = num_labels
        num_corr = int(num_labels * (num_labels - 1) / 2)
        
        if y_train is not None:
            self.scale = np.std(y_train, axis=0)
            self.loc = np.mean(y_train, axis=0)
        else:
            self.scale = 1
            self.loc = 0
        
        self.num_preds = num_mean + num_var + num_corr

    def get_distribution(self, y_pred):
        # y_pred: (num_labels + num_labels + (num_labels - 1) * num_labels / 2)
        #  - first num_labels: mean
        #  - next num_labels: variance
        #  - remaining: correlation components
        
        mu = y_pred[:self.num_labels] + self.loc
        var = jnp.exp(y_pred[self.num_labels:2*self.num_labels]) * self.scale
        L = jnp.tanh(y_pred[2 * self.num_labels:])
        
        corr = jnp.zeros((self.num_labels, self.num_labels))
        idx = 0
        for row in range(self.num_labels):
            for col in range(row + 1, self.num_labels):
                # set correlation matrix
                corr = corr.at[row, col].add(L[idx])
                idx += 1
        
        corr = corr + corr.T + np.eye(self.num_labels)
        cov = var.reshape((self.num_labels, 1)) * var.reshape((1, self.num_labels)) * corr
        
        return mu, cov
    
    def convert_pred(self, y_pred):
        # y_pred: (num_labels + num_labels + (num_labels - 1) * num_labels / 2)
        y_pred = np.reshape(y_pred, (-1, self.num_preds), "F") #[N, num_classes]
        dist_fn = vmap(self.get_distribution, in_axes=(0))
        mu, cov = dist_fn(y_pred)
        return mu, cov

    def loss_fn(self, y_true, y_pred):
        # y_true: (num_labels)
        # y_pred: (num_labels + num_labels + (num_labels - 1) * num_labels / 2)
        #  - first num_labels: mean
        #  - next num_labels: variance
        #  - remaining: correlation components
        
        mu, cov = self.get_distribution(y_pred)
        
        # multi variate gaussian likelihood of y_true given mu, cov
        llh = -0.5 * jnp.log(jnp.linalg.det(cov)) - 0.5 * (y_true - mu).T @ jnp.linalg.inv(cov) @ (y_true - mu)
        loss = -llh
        return loss
    
    def grad_hess(self, y_true, y_pred):
        grad_fn = jax.grad(self.loss_fn, argnums=1)
        hess_fn = jax.jacobian(grad_fn, argnums=1)
        hess_diag_fn = lambda y_true, y_pred: jnp.diag(hess_fn(y_true, y_pred))
        
        grad = vmap(grad_fn, in_axes=(0, 0))(y_true, y_pred)
        hess = vmap(hess_diag_fn, in_axes=(0, 0))(y_true, y_pred)
        return grad, hess
        
    def obj(self, y_true, y_pred):
        
        y_true = np.reshape(y_true, (-1, self.num_preds))
        y_true = y_true[:, :self.num_labels]
        y_pred = np.reshape(y_pred, (y_true.shape[0], self.num_preds)) #[N, num_classes]
        
        grad, hess = self.grad_hess(y_true, y_pred)
        
        # convert from jax.numpy to numpy
        grad = device_get(grad).flatten()
        hess = device_get(hess).flatten()
        
        return grad, hess
    
    def loss(self, y_true, y_pred):
        
        y_true = np.reshape(y_true, (-1, self.num_preds))
        y_true = y_true[:, :self.num_labels]
        y_pred = np.reshape(y_pred, (y_true.shape[0], self.num_preds)) #[N, num_classes]
        
        is_higher_better = False
        vmap_call = vmap(self.loss_fn, in_axes=(0, 0))
        loss = jnp.mean(vmap_call(y_true, y_pred)) / self.num_labels # scal  with num_labels to make it comparable to unit gaussian
                
        return "MVG_loss", loss, is_higher_better
        
def stack_x(x, num_labels):
    num_preds = int(num_labels * (num_labels + 1) / 2) + num_labels
    
    x_stack = np.concatenate([x for _ in range(num_preds)], axis=1)
    return x_stack.reshape((-1, x.shape[1]))

def stack_y(y, num_labels):
    num_preds = int(num_labels * (num_labels + 1) / 2) + num_labels
    num_dupe = int(num_preds / num_labels)
    
    y_stack = np.concatenate([y for _ in range(num_dupe)], axis=1)
    return y_stack.reshape((-1, 1))

if __name__ == '__main__':

    # synthetic data for demonstration
    np.random.seed(0)
    x_train = np.random.randn(1000, 10).astype(np.float32)
    y_train = np.random.randn(1000, 3).astype(np.float32)
    x_test = np.random.randn(200, 10).astype(np.float32)
    y_test = np.random.randn(200, 3).astype(np.float32)

    num_labels = y_train.shape[1]

    x_train_stack = stack_x(x_train, num_labels)
    y_train_stack = stack_y(y_train, num_labels)
    x_test_stack = stack_x(x_test, num_labels)
    y_test_stack = stack_y(y_test, num_labels)

    loss_obj = MultiVariateGaussianLoss(
        num_labels = num_labels,
        y_train = y_train
        )
    model = lgb.LGBMModel(
        objective=loss_obj.obj,
        num_iterations=1000,
        learning_rate=0.01,
    )

    model.fit(
        x_train_stack,
        y_train_stack,
        eval_set=[(x_test_stack, y_test_stack)],
        eval_metric=loss_obj.loss,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(10)],
    )

    mu, cov = loss_obj.convert_pred(model.predict(x_test_stack))
    print("mu:", mu.shape, "cov:", cov.shape)
