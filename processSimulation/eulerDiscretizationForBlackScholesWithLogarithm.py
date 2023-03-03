"""
@author: Andrea Mazzon
"""

import math

from processSimulation.generalProcessSimulation import GeneralProcessSimulation


class EulerDiscretizationForBlackScholesWithLogarithm(GeneralProcessSimulation):
    """
    It provides the simulation of a Black-Scholes process, by simulating its logarithm. In this way we don't have
    discretization error

    Attributes
    ----------
    numberOfSimulations : int
        the number of simulated paths.
    timeStep : float
        the time step of the time discretization.
    finalTime : float
        the final time of the time discretization.
    initialValue : float
        the initial value of the process.
    mySeed : int, optional
        the seed to the generation of the standard normal realizations
    muOfOriginalProcess: float
        the drift of the original process
    sigmaOfOriginalProcess: float
        the log-normal volatility (of the original process)

    Methods
    ----------
    getRealizations():
        It returns all the realizations of the process
    getRealizationsAtGivenTimeIndex(timeIndex):
        It returns the realizations of the process at a given time index
    getAverageRealizationsAtGivenTimeIndex(timeIndex):
        It returns the average realizations of the process at a given time index
    getAverageRealizationsAtGivenTime(time):
        It returns the average realizations of the process at a given time
    getDrift()
        It returns the drift of the logarithm of the Black-Scholes process
    getDiffusion()
        It returns the diffusion of the logarithm of the Black-Scholes process

    """

    def __init__(self, numberOfSimulations, timeStep, finalTime, initialValue, muOfOriginalProcess, sigmaOfOriginalProcess,
                 mySeed=None):
        self.muOfOriginalProcess = muOfOriginalProcess
        self.sigmaOfOriginalProcess = sigmaOfOriginalProcess

        super().__init__(numberOfSimulations, timeStep, finalTime, initialValue, lambda x: math.exp(x),
                         lambda x: math.log(x), mySeed)

    def getDrift(self, currentTime, realizations):
        """
        It returns the drift of the logarithm of the Black-Scholes process
        Parameters
        ----------
        Returns
        -------
        float
            the drift of the logarithm.
        """
        return self.muOfOriginalProcess - 0.5 * self.sigmaOfOriginalProcess ** 2

    def getDiffusion(self, currentTime, realizations):
        """
        It returns the diffusion of the logarithm of the Black-Scholes process
        Parameters
        ----------
        Returns
        -------
        float
            the diffusion of the logarithm.
        """
        return self.sigmaOfOriginalProcess