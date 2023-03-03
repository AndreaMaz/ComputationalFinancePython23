"""
@author: AndreaMazzon
"""

import numpy as np
import math
from random import seed


class GeneralProcessSimulation:
    """
    This is a class whose main goal is to discretize and simulate a continuous time, Ito stochastic process.
    The methods providing drift and diffusion are implemented in sub-classes.

    It is also possible to simulate the process transformed through a given function, and then to transform it back.

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
    functionToBeApplied  : function, optional
        the function that is applied to simulate the process. The default is the identity.
    inverseFunctionToBeApplied : function, optional
        the inverse function that is applied to simulate the process. The default is the identity.
    mySeed : int, optional
        the seed to the generation of the standard normal realizations

    Methods
    ----------
    getRealizations():
        It returns all the realizations of the process
    getRealizationsAtGivenTimeIndex(timeIndex):
        It returns the realizations of the process at a given time index
    getRealizationsAtGivenTimeIndex(time):
        It returns the realizations of the process at a given time
    getAverageRealizationsAtGivenTimeIndex(timeIndex):
        It returns the average realizations of the process at a given time index
    getAverageRealizationsAtGivenTime(time):
        It returns the average realizations of the process at a given time
    getDrift(time, realizations):
        It returns the drift at a given time of the process which is simulated. It gets implemented in derived classes.
    getDiffusion(time, realizations):
        It returns the diffusion at a given time of the process which is simulated. It gets implemented in derived classes.
    """

    def __init__(self, numberOfSimulations, timeStep, finalTime, initialValue,
                 functionToBeApplied=lambda x: x, inverseFunctionToBeApplied=lambda x: x, mySeed=None):
        # note here the use of lambda functions to provide anonymous functions that can be passed as default arguments.
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
        Returns
        -------
        None.
        """
        self.numberOfSimulations = numberOfSimulations
        self.timeStep = timeStep
        self.finalTime = finalTime
        self.initialValue = initialValue
        self.functionToBeApplied = functionToBeApplied
        self.inverseFunctionToBeApplied = inverseFunctionToBeApplied
        self.mySeed = mySeed
        # we generate all the paths for all the simulations
        self.__generateRealizations() #no lazy initialization here

    def __generateRealizations(self):
        # Look at the function vectorize: we can use it to be able to assign an array to a method, or a function, which
        # is supposed to be defined for (arrays of) floats. Another thing: at compilation time, Python does not care about
        # the specific argumets of getDrift and getDiffusion, and does not complain even if they ar enot defined
        # in this specific class.

        vectorizedGetDrift = np.vectorize(self.getDrift)
        vectorizedGetDiffusion = np.vectorize(self.getDiffusion)

        vectorizedFunctionToBeApplied = np.vectorize(self.functionToBeApplied)

        numberOfTimes = math.ceil(self.finalTime / self.timeStep) + 1

        # times on the rows
        self.realizations = np.zeros((numberOfTimes, self.numberOfSimulations))
        self.realizations[0] = [self.inverseFunctionToBeApplied(self.initialValue)] * self.numberOfSimulations

        seed(self.mySeed)#a way to give the seed to be used by numpy.random.standard_normal

        standardNormalRealizations = np.random.standard_normal((numberOfTimes, self.numberOfSimulations))

        # possibly used in order to get the drift and the diffusion
        currentTime = self.timeStep
        for timeIndex in range(1, numberOfTimes):
            pastRealizations = self.realizations[timeIndex - 1]

            self.realizations[timeIndex] = pastRealizations + self.timeStep * vectorizedGetDrift(currentTime, pastRealizations) \
                                           + vectorizedGetDiffusion(currentTime, pastRealizations) \
                                           * math.sqrt(self.timeStep) * standardNormalRealizations[timeIndex]  # the Brownian motion

            currentTime += self.timeStep

        self.realizations = vectorizedFunctionToBeApplied(self.realizations)

    def getRealizations(self):
        """
        It returns all the realizations of the process
        Returns
        -------
        array
            matrix containing the process realizations. The n-th row contains
            the realizations at time t_n
        """
        return self.realizations

    def getRealizationsAtGivenTimeIndex(self, timeIndex):
        """
        It returns the realizations of the process at a given time index
        Parameters
        ----------
        timeIndex : int
             the time index, i.e., the row of the matrix of realizations.
        Returns
        -------
        array
            the vector of the realizations at given time index
        """

        return self.realizations[timeIndex]

    def getRealizationsAtGivenTime(self, time):
        """
        It returns the realizations of the process at a given time
        Parameters
        ----------
        time : float
             the time at which the realizations are returned
        Returns
        -------
        array
            the vector of the realizations at given time
        """

        indexForTime = round(time / self.timeStep)
        return self.realizations[indexForTime]

    def getAverageRealizationsAtGivenTimeIndex(self, timeIndex):
        """
        It returns the average realizations of the process at a given time index
        Parameters
        ----------
        time : int
             the time index, i.e., the row of the matrix of realizations.
        Returns
        -------
        float
            the average of the realizations at given time index
        """

        return np.average(self.realizations(timeIndex))

    def getAverageRealizationsAtGivenTime(self, time):
        """
        It returns the average realizations of the process at a given time
        Parameters
        ----------
        time : int
             the time at which the realizations are returned
        Returns
        -------
        float
            the average of the realizations at given time
        """

        return np.average(self.getRealizationsAtGivenTime(time))