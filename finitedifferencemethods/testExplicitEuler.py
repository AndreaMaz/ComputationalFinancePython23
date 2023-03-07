"""
Here we test the Explicit Euler scheme and the Upwind method, computing the price of a call option. We investigate their
stability.

@author: Andrea Mazzon
"""
import numpy as np
import math

from explicitEuler import ExplicitEuler


strike = 2
payoff = lambda x : np.maximum(x - strike, 0)

functionLeft = lambda x, t : 0

#This is because from put-call parity and since the price of a put is close to zero for large values of the underyling,
#the price of the call for large values x of the underlying can be approximated by x - strike * math.exp(-r * T),
#where T is the maturity
functionRight = lambda x, t : x - strike * math.exp(-r * t)

dx = 0.1
xmin = 0
xmax = 7

sigma = 0.5
sigmaFunction = lambda x : sigma
r = 0.8

dt = dx*dx /(sigma*xmax)**2

tmax = 2


solver = ExplicitEuler(dx, dt, xmin, xmax, tmax, r, sigmaFunction, payoff, functionLeft, functionRight)

solver.solveAndPlot()

