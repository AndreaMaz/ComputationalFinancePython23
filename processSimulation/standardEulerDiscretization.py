"""
@author: andreamazzon
"""

from processSimulation.generalProcessSimulation import GeneralProcessSimulation


class StandardEulerDiscretization(GeneralProcessSimulation):
    """
    It provides the Euler disacretization and simulation of a local volatility process.

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
    functionToBeApplied : function, optional
        the function f that is applied to simulate the process. The default is the identity.
    inverseFunctionToBeApplied : function, optional
        the inverse function f^(-1) that is applied to simulate the process. The default is the identity.
    mySeed : int, optional
        the seed to the generation of the standard normal realizations
    muFunction: float
        the function for the drift of the process Y = f(X). It is a function of time and space.
    sigmaFunction: float
        the function for the volatility  of the process Y = f(X). It is a function of time and space.

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
    getDrift(time, realizations)
        It returns the drift mu(t,Y_t) of the process Y_t = f(X_t) at time t
    getDiffusion(time, realizations)
        It returns the diffusion sigma(t,Y_t) of the process Y_t = f(X_t) at time t

    """

    def __init__(self, numberOfSimulations, timeStep, finalTime, initialValue, muFunction, sigmaFunction,
                 functionToBeApplied=lambda x: x, inverseFunctionToBeApplied=lambda x: x, mySeed=None):
        """
        Parameters
        ----------
        numberOfSimulations : int
            the number of simulated paths.
        timeStep : float
            the time step of the time discretization.
        finalTime : float
            the final time of the time discretization.
        initialValue : float
            the initial value of the process.
        functionToBeApplied : function, optional
            the function that is applied to simulate the process. The default is the identity.
        inverseFunctionToBeApplied : function, optional
            the inverse function that is applied to simulate the process. The default is the identity.
        mySeed : int, optional
            the seed to the generation of the standard normal realizations
        muFunction: float
            the function for the drift
        sigmaFunction: float
            the function for the volatility
        """

        self.muFunction = muFunction
        self.sigmaFunction = sigmaFunction

        super().__init__(numberOfSimulations, timeStep, finalTime, initialValue, functionToBeApplied,
                         inverseFunctionToBeApplied, mySeed)

    def getDrift(self, time, realization):
        """
        It returns the drift mu(t,Y_t) of the process Y_t = f(X_t) at time t
        Parameters
        ----------
        time : double
            the time.
        realizations : double
            the realizations of the process.
        Returns
        -------
        float
            the drift of the process.
        """
        return self.muFunction(time, realization)

    def getDiffusion(self, time, realization):
        """
        It returns the diffusion sigma(t,Y_t) of the process Y_t = f(X_t) at time t
        Parameters
        ----------
        time : double
            the time.
        realizations : double
            the realizations of the process.
        Returns
        -------
        float
            the diffusion of the process.
        """
        return self.sigmaFunction(time, realization)