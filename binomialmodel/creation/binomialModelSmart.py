"""
@author: Andrea Mazzon
"""
import numpy as np
import scipy.special
import math
from binomialmodel.creation.binomialModel import BinomialModel


class BinomialModelSmart(BinomialModel):
    """
    In this class we implement the generation of a binomial model by using an approach that is computationally more
    efficient than Monte-Carlo, since it needs much less memory, and most of all is not prone to the simulation problems
    that Monte Carlo exhibits.

    In particular, here we don't rely on pseudo random numbers generations, but at a given time we just consider ALL the
    possible realizations of the process attributing a (analytic!) probability to every realization. We then compute
    the average via the weighted sum of the realizations with their probabilities.

    The realizations of the process are stored in an array matrix.
    ...

    Attributes
    ----------
    initialValue : float
        the initial value S(0) of the process
    decreaseIfDown : float
        the number d such that S(j+1)=S(j) with probability 1 - q
    increaseIfUp : float
        the number u such that S(j+1)=S(j) with probability q
    numberOfTimes : int
        the number of times at which the process is simulated, initial time included
    interest rate : float
        the interest rate rho such that the risk free asset B follows the dynamics B(j+1) = (1+rho)B(j)
    riskNeutralProbabilityUp : float
        the risk neutral probability q =(1+rho-d)/(u-d) such that
        P(S(j+1)=S(j)*u) = q, P(S(j+1)=S(j)*d) = 1 - q,
        u > rho+1, d<1
    realizations : array
        a matrix containing the realizations of the process


    Methods
    -------

    generateRealizations()
        It generates and returns the matrix representing all the possible  realizations of the process up to time
        self.numberOfTimes - 1.
    getRealizations()
        It returns the realizations of the process.
    getRealizationsAtGivenTime(timeIndex)
        It returns the realizations of the process at time timeIndex
    getDiscountedAverageAtGivenTime(timeIndex)
        It returns the average of the process at time timeIndex discouted at time 0
    getEvolutionDiscountedAverage()
        It returns the evolution of the average of the process discounted at time 0
    printEvolutionDiscountedAverage()
        It prints the evolution of the average of the process discounted at time 0
    plotEvolutionDiscountedAverage()
        It plots the evolution of the average of the process discounted at time 0
    getPercentageOfGainAtGivenTime(timeIndex)
        It returns the percentage probability that (1+rho)^(-j)S(j)>S(0)
    getEvolutionPercentageOfGain()
        It returns the evolution of the percentage probability that (1+rho)^(-j)S(j)>S(0), for j going from 1 to
        self.numberOfTimes - 1
    printEvolutionPercentagesOfGain()
        It prints the evolution of the percentage probability that (1+rho)^(-j)S(j)>S(0), , for j going from 1 to
        self.numberOfTimes - 1
    plotEvolutionPercentagesOfGain()
        It plots the evolution of the percentage probability that (1+rho)^(-j)S(j)>S(0), , for j going from 1 to
        self.numberOfTimes - 1
    getProbabilitiesOfRealizationsAtGivenTime(timeIndex)
        It returns the probabilities corresponding to every possible realization of the process at time timeIndex.
    printProbabilitiesOfRealizationsAtGivenTime(timeIndex)
        It prints the probabilities corresponding to every possible realization of the process at time timeIndex.
   findThreshold(timeIndex):
        It returns the smallest integer k such that (u)^kd^(timeIndex-k) > (1+rho)^n, i.e., such that the realization of the
        process given by k ups and timeIndex - k downs, discounted at time 0, is bigger than the initial value.



    """

    def __init__(self, initialValue, decreaseIfDown, increaseIfUp, numberOfTimes, interestRate=0):
        """
        Attributes
        ----------
        initialValue : float
            the initial value S(0) of the process
        decreaseIfDown : float
            the number d such that S(j+1)=S(j) with probability 1 - q
        increaseIfUp : float
            the number u such that S(j+1)=S(j) with probability q
        numberOfTimes : int
            the number of times at which the process is simulated, initial time included
        interest rate : double
            the interest rate rho such that the risk free asset B follows the dynamics B(j+1) = (1+rho)B(j)
        """
        super().__init__(initialValue, decreaseIfDown, increaseIfUp, numberOfTimes, interestRate)

    def generateRealizations(self):
        """
        It generates all the realizations of the process up to time self.numberOfTimes - 1.

        At every time N, there are N+1 possible reaalizations, depending on the number of times the process goes up:
        the "first" realization is given by N "ups", the second by N-1 ups and 1 down, etc.

        The realizations are stored in a matrix, which is not "full" but basically made by a diagonal and one triangle.

        Returns
        -------
        realizations : array
            a matrix storing all the possible realizations of the process up to time self.numberOfTimes - 1.
        """
        # at every time N, there are N+1 possible values. The final time is self.numberOfTimes - 1
        realizations = np.full((self.numberOfTimes, self.numberOfTimes), math.nan)
        realizations[0, 0] = self.initialValue
        for k in range(1, self.numberOfTimes):
            # the first realization is the previous first realization times u
            realizations[k, 0] = self.increaseIfUp * realizations[k - 1, 0]
            # the second is the previous first realization times d, and so on up to the last one, which is the previous
            # last one times d
            realizationsAtCurrentTime = self.decreaseIfDown * realizations[k - 1, 0:k]
            realizations[k, 1:k + 1] = realizationsAtCurrentTime
        return realizations

    def getRealizationsAtGivenTime(self, timeIndex):
        """
        It returns all the realizations of the process at time timeIndex

        Note that the first realization of the list is the one corresponding to all ups and no downs.

        Parameters
        ----------
        timeIndex : int
            the time at which we want the realizations of the process for all the simulations

        Returns
        -------
        array
            a vector representing the realizations of the process at time timeIndex.

        """

        # we don't return all the vector of the matrix, but only the first N realizations, with N = timeIndex
        return self.realizations[timeIndex, 0:timeIndex + 1]

    def getProbabilitiesOfRealizationsAtGivenTime(self, timeIndex):
        """
        It returns the probabilities corresponding to every possible realization of the process at time timeIndex.

        Note that the first realization of the list is the one corresponding to all ups and no downs.

        The probabilities are computed using the fact that the realizations have Bernoulli distribution: in particular,
        the probability of the realization with k ups and N - k downs is
        N!/(k!(n-k)!)q^k(1-q)^(N-k),
        where q = self.riskNeutralProbabilityUp

        Parameters
        ----------
        timeIndex : int
            the time at which we want the probabilities.

        Returns
        -------
        array
            a vector representing the probabilities of every possible realization of the process at time timeIndex.

        """
        if timeIndex == 0:
            return 1.0
        else:
            q = self.riskNeutralProbabilityUp

            probabilities = [scipy.special.binom(timeIndex, numberOfDowns) * q ** (timeIndex - numberOfDowns)
                             * (1 - q) ** numberOfDowns for numberOfDowns in range(timeIndex + 1)]
            return probabilities

    def printProbabilitiesOfRealizationsAtGivenTime(self, timeIndex):
        """
        It prints the probabilities corresponding to every possible realization of the process at time timeIndex.

        The probabilities are computed using the fact that the realizations have Bernoulli distribution: in particular,
        the probability of the realization with k ups and N - k downs is
        N!/(k!(n-k)!)q^k(1-q)^(N-k),
        where q = self.riskNeutralProbabilityUp

        Parameters
        ----------
        timeIndex : the time at which we want to print the probabilities

        Returns
        -------
        None.

        """
        probabilities = self.getProbabilitiesOfRealizationsAtGivenTime(timeIndex)
        print("The probabilities of the realizations at time ", timeIndex, " from the largest realizations to the smallest are ")
        print('\n'.join('{:.3}'.format(prob) for prob in probabilities))


    def getDiscountedAverageAtGivenTime(self, timeIndex):
        """
        Parameters
        ----------
        timeIndex : int
            The time at which we want the average of the realizations of the process, discounted at time 0.

        Returns
        -------
        float
            the average of the realizations of the process at time timeIndex, discounted at time 0.

        """
        if timeIndex == 0:
            return self.initialValue
        else:
            realizations = self.getRealizationsAtGivenTime(timeIndex)
            probabilities = self.getProbabilitiesOfRealizationsAtGivenTime(timeIndex)
            # we discount the weighted sum of the realizations
            discountedAverage = (1 + self.interestRate) ** (-timeIndex) * np.dot(probabilities, realizations)
            return discountedAverage


    def findThreshold(self, timeIndex):
        """
        It returns the smallest integer k such that (u)^kd^(timeIndex-k) > (1+rho)^N, i.e., such that the realization of the
        process given by k ups and timeIndex - k downs, discounted at time 0, is bigger than the initial value.

        Parameters
        ----------
        timeIndex : int
            the time at which we want to compute such a threshold.

        Returns
        -------
        int
            the smallest integer k such that (u)^kd^(timeIndex-k) > (1+rho)^N

        """
        rho = self.interestRate
        u = self.increaseIfUp
        d = self.decreaseIfDown
        return math.ceil(math.log(((1 + rho) / d) ** timeIndex, u / d))

    def getPercentageOfGainAtGivenTime(self, timeIndex):
        """
        Parameters
        ----------
        timeIndex : int
            The time j at which we want the probability that (1+rho)^(-j)S(j)>S(0) with rho = self.interestRate

        Returns
        -------
        float
            the probability that (1+rho)^(-timeIndex)S(timeIndex)>S(0) with rho = self.interestRate

        """
        if timeIndex == 0:
            return 100.0
        else:
            probabilities = self.getProbabilitiesOfRealizationsAtGivenTime(timeIndex)
            threshold = self.findThreshold(timeIndex)
            # we sum the probabilities corresponding to all the realizations with enough ups k, and then we take the
            # percentage probability. We have + 1 because 0:n is 0,1,..,n-1. We want to consider all the
            # realizations with a number of downs <= timeIndex - threshold
            return 100.0 * sum(probabilities[0:timeIndex - threshold + 1])

