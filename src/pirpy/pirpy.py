import logging
from reservoirpy import Model
from sklearn.base import BaseEstimator
from scipy.optimize import minimize
from typing import Callable
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

def add_knowledge(esn: Model, scaler: BaseEstimator, dt: float,
                  dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray],
                  X_pred: np.ndarray, y_pred: np.ndarray, warmup: int,
                  t0: float = 0, alpha: float = 0.5, method = 'L-BFGS-B', verbose=True, **options):
    """
    Add physical knowledge to the readout of an Echo State Network using scipy.optimize.minimize.

    Parameters
    ----------
    esn : Model
        ReservoirPy model. The last node must be a readout node (with a Wout matrix and optionally a bias).
        ESNs different from ReservoirPy's implementation could be used, but they must have a 'run' methond
        and 'nodes' attribute that mimics the behavior of ReservoirPy's Model class.

    scaler : BaseEstimator
        Scikit-learn's scaler object used to normalize the target data.
    
    dt : float
        Time step of the Echo State Network.

    dynamics : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Function that computes the derivatives of the output data.
        It must have the signature f(t: np.ndarray ,y: np.ndarray) -> dy: np.ndarray,
        like the functions expected by scipy.integrate.solve_ivp.
    
    X_pred : np.ndarray
        Input data during prediction horizon.
    
    y_pred : np.ndarray
        Output data during prediction horizon.
    
    warmup : int
        Number of samples to warmup the reservoir before prediction.
    
    t0 : float, optional
        Initial time of the prediction horizon. Default is 0.
    
    alpha : float, optional
        Weighting factor between prediction error and physical consistency error.
        Should be in the range [0, 1]. Default is 0.5 (equally weighted).

    method : str, optional
        Scipy's optimization method. Default is 'L-BFGS-B'.

    verbose : bool, optional
        If True, print the optimization progress. Default is True.

    options : dict, optional
        SciPy's optimization options. See SciPy's documentation for more information.
    """
    
    # Status function in order to see the optimization progress
    Niter = 1
    def _status(*, intermediate_result):
        nonlocal Niter
        msg = f'[Niter {Niter}][{datetime.now()}] Loss: {intermediate_result.fun}'
        if verbose:
            logger.status(msg)
        else:
            logger.info(msg)
        Niter += 1

    # Prepare data
    X_pred, y_pred = _prepare_data(X_pred, y_pred)

    # Time vector during the prediction horizon
    t = np.arange(t0, t0 + len(X_pred)*dt, dt)

    # Arguments for the objective function
    Wout = esn.nodes[-1].Wout.flatten()
    args = (esn, X_pred, y_pred, t, dynamics, scaler, warmup, alpha, dt)

    # Optimization
    logger.status(f'[Start][{datetime.now()}] Loss: {objective(Wout, *args)}')
    opt_res = minimize(objective, Wout, args=args, method=method, options=options, callback=_status)
    logger.status(f'[End][{datetime.now()}] Loss: {opt_res.fun} - [{datetime.now()}]')

    # Last update of the Wout matrix
    _update(esn, opt_res.x)

def objective(Wout: np.ndarray, *args) -> float:
    """
    Objective function to minimize. It computes the prediction error and the physical consistency error.

    Direct implementation of the methodology presented in https://doi.org/10.1016/j.jocs.2020.101237,
    taking into account ReservoirPy's implementation of Echo State Networks.

    The function has the signature required by the scipy.optimize.minimize function.

    Parameters
    ----------
    Wout : np.ndarray
        Weight matrix of the readout node. The only parameter to optimize.

    args : tuple
        Tuple with some important but fixed arguments. The tuple must contain the following elements:
        - esn: ReservoirPy's Model object.
        - dt: Echo State Network's time step.
        - X_pred: Input data during prediction horizon.
        - y_pred: Output data during prediction horizon.
        - t: Time vector during the prediction horizon.
        - dynamics: Function that computes the derivatives of the output data.
                    It must have the signature f(t: np.ndarray ,y: np.ndarray) -> dy: np.ndarray,
                    like the functions expected by scipy.integrate.solve_ivp.
        - scaler: Scikit-learn's scaler object used to normalize the tarjet data.
        - warmup: Number of samples to warmup the reservoir before prediction.
        - alpha: Weighting factor between prediction error and physical consistency error.
                 Should be in the range [0, 1].
    """

    esn, dt, X_pred, y_pred, t, dynamics, scaler, warmup, alpha, *_  = args

    # Update readout with optimized Wout
    _update(esn, Wout)

    # Warmup
    esn.run(X_pred[:warmup])

    # Compute all predicted outputs
    y_hat = esn.run(X_pred[warmup:])

    # Compute all predicted derivatives with forward Euler
    y_diff_hat = np.diff(y_hat, axis=0, prepend=np.atleast_2d(X_pred[warmup])) / dt

    # Compute the derivatives according to the dynamics
    y_dot = dynamics(t, scaler.inverse_transform(y_hat).T).T * scaler.scale_

    # Sanitize derivatives taking into account the possibility of partial knowledge
    y_dot, y_diff_hat = _check_derivatives(y_dot, y_diff_hat)

    # F = y_dot - N(y)
    F = y_dot - y_diff_hat

    # Compute prediction error and physical consistency error
    Ed = np.mean(np.mean((y_hat - y_pred[warmup:])**2, axis=0))
    Ep = np.mean(np.linalg.norm(F, axis=1)**2)

    return (1-alpha)*Ed + alpha*Ep

def _prepare_data(X, y):
    """
    Data is expected to be in the shape (timesteps, n_features), and it is assumed that the number of timesteps is higher than the number of features.
    """
    X, y = np.atleast_2d(X), np.atleast_2d(y)
    if X.shape[0] < X.shape[1]:
        X = X.T
    if y.shape[0] < y.shape[1]:
        y = y.T
    return X, y

def _update(esn, Wout):
    """
    Update the Wout matrix in the readout node of the ESN, reshaping it if necessary.
    """
    Wout_outdated = esn.nodes[-1].Wout
    esn.nodes[-1].Wout = Wout.reshape(Wout_outdated.shape) if Wout.shape != Wout_outdated.shape else Wout

def _check_derivatives(y_dot, y_diff_hat):
    """
    Check if all columns in y_dot are well populated. Our physical knowledge could be partial, and some derivatives could be missing.
    Return only the columns that are well populated.
    """
    mask = np.all(np.isfinite(y_dot), axis=0)
    return y_dot[:, mask], y_diff_hat[:, mask]