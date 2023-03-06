#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""
import numpy as np
import math


class KnockOutOption:
    """
    The main goal of this class is to give the value of a Knock-out option with a general payoff at a given maturity.
    ...

    Attributes
    ----------
    underlyingModel : binomial.creation.BinomialModel the underlying binomialmodel


    Methods
    -------

    getValuesOptionBackward(payoffFunction, maturity)
        It returns the values for every time until maturity of the option with given maturity.
    getValuesOptionBackwardAtGivenTime(payoffFunction, currentTime, finalTime):
        It returns the values at currentTime of the option with given maturity.
    getValuesDiscountedOptionBackwardAtGivenTime(payoffFunction, currentTime, finalTime):
        It returns the discounted values at currentTime of the option with given maturity.
    getInitialValueOption(payoffFunction, maturity):
        It returns the initial value of the option with given maturity.
    getInitialDiscountedValueOption(payoffFunction, maturity):
        It returns the initial diascounted value of the option with given maturity.
    getStrategy(self, payoffFunction, maturity):
        It returns two matrices, describing how much money must be invested in the risk free and in the risky asset at
        any time before maturity in order to replicate the payoff at maturity
     getStrategyAtGivenTime(self, payoffFunction, currentTime, maturity):
        It returns two vectors, describing how much money must be invested in the risk free and in the risky asset at
        currentTime in order to replicate the payoff at maturity
    """

    def __init__(self, underlyingModel):
        """

        Parameters
        ----------
        underlyingModel : binomial.creation.BinomialModel
            the underlying binomial model
        """
        self.underlyingModel = underlyingModel

    def getValuesOptionBackward(self, payoffFunction, maturity, lowerBarrier = -np.inf, upperBarrier = np.inf):
        """
        It returns the values for every time until maturity of the option with given maturity.

        Note that the values are given as a triangular matrix, since at time k we have k + 1 value of the Option.

        Parameters
        ----------
        payoffFunction : lambda function
            the function representing the payoff we want to valuate.
        maturity : int
            the time at which the Option has to replicate the payoff
        lowerBarrier : float
            the lower barrier of the option
        upperBarrier : float
            the upper barrier of the option

        Returns
        -------
        valuesOption : array
            a (triangular) matrix describing the value of the Option at
            every time before maturity

        """
        binomialModel = self.underlyingModel
        q = binomialModel.riskNeutralProbabilityUp

        # we consider a number of times equal to maturity + 1
        valuesOption = np.full((maturity + 1, maturity + 1), math.nan)

        # realizations of the process at maturity
        processRealizations = binomialModel.getRealizationsAtGivenTime(maturity)

        # payoffs at maturity
        payoffRealizations = [payoffFunction(x) if x > lowerBarrier and x < upperBarrier else 0  for x in processRealizations]
        # the final values of the Option are simply the payoffs
        valuesOption[maturity, :] = payoffRealizations

        for timeIndexBackward in range(maturity - 1, -1, -1):
            processRealizations = binomialModel.getRealizationsAtGivenTime(timeIndexBackward)
            # V(k,j)=qV(k+1,j+1)+(1-q)V(k,j+1), with j current time, k number of ups until current time,
            # ONLY IF the realization with j ups at time k is between the two barriers
            valuesOption[timeIndexBackward, 0: timeIndexBackward + 1] = \
                (q * valuesOption[timeIndexBackward + 1, 0:timeIndexBackward + 1] + \
                 (1 - q) * valuesOption[timeIndexBackward + 1, 1:timeIndexBackward + 2]) \
                * (processRealizations < upperBarrier).astype(int) * (processRealizations > lowerBarrier).astype(int)

        return valuesOption

    def getValuesOptionBackwardAtGivenTime(self, payoffFunction, currentTime, maturity,
                                              lowerBarrier=-np.inf, upperBarrier=np.inf):
        """
        It returns the values at currentTime of the option with given maturity.

        Parameters
        ----------
        payoffFunction : lambda function
            the function representing the payoff we want to valuate.
        currentTime : int
            the time at which we want the value of the Option
        maturity : int
            the time at which the Option has to replicate the payoff
        lowerBarrier : float
            the lower barrier of the option
        upperBarrier : float
            the upper barrier of the option

        Returns
        -------
        valuesOptionAtCurrentTime : array
            a vector describing the values of the Option at currentTime

        """
        allValuesOption = self.getValuesOptionBackward(payoffFunction, maturity, lowerBarrier, upperBarrier)
        valuesOptionAtCurrentTime = allValuesOption[currentTime, 0: currentTime + 1]
        return valuesOptionAtCurrentTime

    def getValuesDiscountedOptionBackwardAtGivenTime(self, payoffFunction, currentTime, maturity,
                                                        lowerBarrier=-np.inf, upperBarrier=np.inf):
        """
        It returns the discounted values at currentTime of a Option
        replicating the payoff of the option at maturity.

        Parameters
        ----------
        payoffFunction : lambda function
            the function representing the payoff we want to valuate.
        currentTime : int
            the time at which we want the discounted value of the Option
        maturity : int
            the time at which the Option has to replicate the payoff
        lowerBarrier : float
            the lower barrier of the option
        upperBarrier : float
            the upper barrier of the option

        Returns
        -------
        valuesOptionAtCurrentTime : array
            a vector describing the discounted values of the Option at currentTime

        """

        binomialModel = self.underlyingModel
        rho = binomialModel.interestRate

        # Note that here we can multiply directly the vector by the float value.
        # This is not the case for lists
        discountedValuesOptionAtCurrentTime = \
            self.getValuesOptionBackwardAtGivenTime(payoffFunction, currentTime, maturity, lowerBarrier, upperBarrier) \
            * ((1 + rho) ** (-(maturity - currentTime)))

        return discountedValuesOptionAtCurrentTime

    def getInitialValueOption(self, payoffFunction, maturity,
                                 lowerBarrier=-np.inf, upperBarrier=np.inf):
        """
        It returns the initial value of a Option replicating the payoff of
        the option at maturity.

        This should be equal to the value returned by
        evaluateDiscountedPayoff(payoffFunction, maturity):

        Parameters
        ----------
        payoffFunction : lambda function
            the function representing the payoff we want to valuate.
        maturity : int
            the time at which the Option has to replicate the payoff
        lowerBarrier : float
            the lower barrier of the option
        upperBarrier : float
            the upper barrier of the option

        Returns
        -------
        initialValueOption : float
            the discounted value of the Option at initial time
        """

        OptionValues = self.getValuesOptionBackward(payoffFunction, maturity, lowerBarrier, upperBarrier)
        initialValueOption = OptionValues[0, 0]

        return initialValueOption

    def getInitialDiscountedValueOption(self, payoffFunction, maturity,
                                           lowerBarrier=-np.inf, upperBarrier=np.inf):
        """
        It returns the discounted initial value of a Option replicating the payoff of
        the option at maturity.


        Parameters
        ----------
        payoffFunction : lambda function
            the function representing the payoff we want to valuate.
        maturity : int
            the time at which the Option has to replicate the payoff
        lowerBarrier : float
            the lower barrier of the option
        upperBarrier : float
            the upper barrier of the option

        Returns
        -------
        initialDiscountedValueOption : float
            the discounted value of the Option at initial time
        """

        binomialModel = self.underlyingModel
        rho = binomialModel.interestRate

        initialDiscountedValueOption = self.getInitialValueOption(payoffFunction, maturity, lowerBarrier, upperBarrier) \
                                          * (1 + rho) ** (-maturity)
        return initialDiscountedValueOption


