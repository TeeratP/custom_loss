import numpy as np
import lightgbm as lgb
import jax.numpy as jnp
from functools import partial
import jax
from jax import grad, jit, vmap, value_and_grad, jacobian

class MVGModel:
    def __init__(self, num_labels, weight_matrix = None, **kwargs):
        
        self.model_kwargs = kwargs
        self.num_labels = num_labels
        self.num_pred_vars = num_labels + num_labels * (num_labels + 1) // 2
        self.weight_matrix = weight_matrix if weight_matrix is not None else np.eye(num_labels)
        assert self.weight_matrix.shape[1] == num_labels, "Weight matrix must have same number of columns as number of labels"
        
        self.model = None
    
    def get_dist(self, y_pred_row):
        # y_pred: [self.num_pred_vars]
        # - first num_labels: mean
        # - next num_labels: variance
        # - remaining: correlation components
        
        mu = y_pred_row[:self.num_labels] + self.y_mean
        std = jnp.exp(y_pred_row[self.num_labels:2*self.num_labels]) * self.y_std
        L = jnp.tanh(y_pred_row[2 * self.num_labels:])
        
        corr = jnp.zeros((self.num_labels, self.num_labels))
        idx = 0
        for row in range(self.num_labels):
            for col in range(row + 1, self.num_labels):
                # set correlation matrix
                corr = corr.at[row, col].add(L[idx])
                idx += 1
        
        corr = corr + corr.T + np.eye(self.num_labels)
        cov = std.reshape((self.num_labels, 1)) * std.reshape((1, self.num_labels)) * corr
        return mu, cov
    
    def get_obj(self, y_true):
        
        # y_true: [num_rows, num_labels]
        
        # initialize the mean and std of predicted distribution (due to complexity of the objective function, model might not converge to answer given low number of iterations)
        self.y_mean = y_true.mean(axis = 0)
        self.y_std = y_true.std(axis = 0)
        
        def loss_fn_row(y_true_row, y_pred_row):
            # y_true_row: [num_labels]
            # y_pred_row: [self.num_pred_vars]
            # self.weight_matrix: [num_weights, num_labels]
            
            mu, cov = self.get_dist(y_pred_row) # [num_labels], [num_labels, num_labels]
            
            y_true_row = jnp.expand_dims(y_true_row, axis = 0) @ self.weight_matrix.T # [1, num_weights]
            y_true_row = y_true_row[0, :] # [num_weights]
            
            mu = jnp.expand_dims(mu, axis = 0) @ self.weight_matrix.T # [1, num_weights]
            mu = mu[0, :] # [num_weights]
            cov = self.weight_matrix @ cov @ self.weight_matrix.T # [num_weights, num_weights]
            
            llh = jax.scipy.stats.multivariate_normal.logpdf(y_true_row, mu, cov)
            loss = -llh
            return loss
        
        # get loss function, gradient, and hessian (diagonal only) for each row
        # all with shape [self.num_pred_vars]
        grad_fn_row = grad(loss_fn_row, argnums = 1) # gradient with respect to y_pred_row
        hess_fn_row = jacobian(grad_fn_row, argnums = 1) # hessian with respect to y_pred_row
        hess_diag_fn_row = lambda y_true_row, y_pred_row: jnp.diag(hess_fn_row(y_true_row, y_pred_row)) # diagonal of hessian with respect to y_pred_row
        
        self.loss_fn = lambda y_pred: vmap(loss_fn_row, in_axes = (0, 0))(y_true, y_pred)
        self.grad_fn = lambda y_pred: vmap(grad_fn_row, in_axes = (0, 0))(y_true, y_pred)
        self.hess_fn = lambda y_pred: vmap(hess_diag_fn_row, in_axes = (0, 0))(y_true, y_pred)
        
    def fit(self, x, y, x_val = None, y_val = None, **kwargs):
        
        self.get_obj(y)
        
        @jit
        def grad_hess(y_true, y_pred): # don't use y_true since it is dummy y
            y_pred = jnp.reshape(y_pred, (y.shape[0], self.num_pred_vars), "F")
            grad = self.grad_fn(y_pred)
            hess = self.hess_fn(y_pred)
            
            #convert jnp to np
            grad = grad.flatten("F")
            hess = hess.flatten("F")
            
            return grad, hess
        
        def obj_fn(y_true, y_pred):
            grad, hess = grad_hess(y_true, y_pred)
            return np.asarray(grad), np.asarray(hess)
        
        @jit
        def loss_fn(y_true, y_pred):
            return self.loss_fn(yp).mean()
        
        def eval_fn(y_true, y_pred): # don't use y_true since it is dummy y
            y_pred = jnp.reshape(y_pred, (y.shape[0], self.num_pred_vars), "F")
            loss = loss_fn(y, y_pred)
            return ("loss", np.asarray(loss), False)
        
        y_dummy = np.zeros((x.shape[0]))
        
        self.model = lgb.LGBMModel(
            objective = obj_fn,
            num_classes = self.num_pred_vars, 
            **self.model_kwargs)
        
        if x_val is not None and y_val is not None:
            self.model.fit(x, y_dummy, 
                    eval_set = [(x_val, y_dummy)],
                    eval_metric = eval_fn,
                    **self.model_kwargs)
        else:
            self.model.fit(x, y_dummy, 
                        #    eval_metric = eval_fn,
                        **kwargs)
    
    def predict(self, x):
        
        assert self.model is not None, "Model not fitted yet"
        
        var_preds = self.model.predict(x)
        var_preds = jnp.reshape(var_preds, (x.shape[0], self.num_pred_vars), "F")
        get_dists = vmap(self.get_dist, in_axes = (0))
        mu, cov = get_dists(var_preds) # [num_rows, num_labels], [num_rows, num_labels, num_labels]
        return np.asarray(mu), np.asarray(cov)
    
if __name__ == '__main__':
    
    # sample usage of MVGModel
    # from functions.mvgmodel import MVGModel
    
    # generate data
    np.random.seed(0)
    num_rows = 10000
    num_features = 10
    num_labels = 3
    
    x = np.random.randn(num_rows, num_features)
    y = np.random.randn(num_rows, num_labels)
    
    # fit model
    model = MVGModel(num_labels)
    model.fit(x, y)