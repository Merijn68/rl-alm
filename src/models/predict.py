# Time series prediciton models
import numpy as np


def vasicek(r0, mu, sigma, dt):
    return r0 + mu * dt + sigma * np.sqrt(dt) * np.random.normal()


#   return r0 + mu * dt + 2520 * sigma * np.random.normal()

# def vasicek(r0, mu, sigma, dt, seed=777):
#    np.random.seed(seed)
#    return (mu-r0)*dt + sigma*np.random.normal()
