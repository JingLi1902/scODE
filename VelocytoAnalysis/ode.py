import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

def denoise_data(data, method='median', window_size=3):
    """
    Apply a denoising algorithm to the data.
    method: 'median' (median filter) or 'mean' (mean filter)
    window_size: Size of the window for the filter
    """
    if method == 'median':
        from scipy.ndimage import median_filter
        return median_filter(data, size=window_size)
    elif method == 'mean':
        from scipy.ndimage import uniform_filter
        return uniform_filter(data, size=window_size)
    else:
        raise ValueError("Unknown denoising method: {}".format(method))

def runge_kutta_step(f, y, t, dt):
    """
    Perform a single step of the 4th order Runge-Kutta method.
    f: Function representing the ODE (dy/dt = f(y, t))
    y: Current value of the variable
    t: Current time
    dt: Time step
    """
    k1 = f(y, t)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(y + dt * k3, t + dt)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def ode_function(model, xt, tf):
    if tf is not None:
        return model.predict(xt[:, tf])
    else:
        return model.predict(xt)

def ODE_SIMULATION(counts, model, genes=None, tfs=None, dt=0.02, n=100, m=1, noise=None, noise_method='median', rk4=False):
    '''Run ODE Simulation'''
    if genes is None:
        genes = np.array([True] * counts.shape[1])
    x = counts[:, genes]
    
    if tfs is not None:
        tf = np.array([list(np.arange(genes.shape[0])[genes + tfs]).index(g) for g in list(np.arange(genes.shape[0])[tfs])])
    else:
        tf = None

    # Denoise the data if noise is not None
    if noise is not None:
        x = denoise_data(x, method=noise_method)

    # Simulation
    xt = x
    path = []
    path.append(xt)
    
    for i in range(n):
        if rk4:
            xt = runge_kutta_step(lambda y, t: ode_function(model, y, tf), xt, i * dt, dt)
        else:
            if tf is not None:
                vt = model.predict(xt[:, tf])
            else:
                vt = model.predict(xt)
            if noise is not None:
                vt += np.random.normal(loc=0, scale=noise, size=xt.shape)
            xt = xt + dt * vt

        xt[xt < 0] = 0
        
        if np.mod(i + 1, m) == 0:
            path.append(xt)
    
    print('ODE Simulation Done.')
    return np.array(path)
