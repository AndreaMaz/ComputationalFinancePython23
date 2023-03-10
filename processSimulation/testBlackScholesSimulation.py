
"""
In this class we do some first tests for the use of Monte-Carlo methods to 
simulate continuous time stochastic processes.

@author: Andrea Mazzon
"""
import numpy as np
import math

from eulerDiscretizationForBlackScholesWithLogarithm import EulerDiscretizationForBlackScholesWithLogarithm
from standardEulerDiscretization import StandardEulerDiscretization

from analyticformulas.analyticFormulas import blackScholesPriceCall


numberOfSimulations = 100
timeStep = 0.2
finalTime = 2

initialValue = 150
r = 0.05
sigma = 0.7


maturity = finalTime
strike = initialValue


payoffFunction = lambda x : np.maximum(x-strike,0)

analyticPrice = blackScholesPriceCall(initialValue, r, sigma, maturity, strike)

errorsWithStandard = []

errorsWithLogarithm = []

numberOfTests = 100

for k in range(numberOfTests):

    #price and error generating the process by simulating the logarithm 
    
    eulerBlackScholes= EulerDiscretizationForBlackScholesWithLogarithm(numberOfSimulations, timeStep, finalTime,
                       initialValue, r, sigma, mySeed = 10)
    
    processRealizationsWithLogarithm = eulerBlackScholes.getRealizationsAtGivenTime(maturity)

    payoffsWithLogarithm = payoffFunction(processRealizationsWithLogarithm)
    priceWithLogarithm = math.exp(-r*maturity) * np.mean(payoffsWithLogarithm)


    errorsWithLogarithm.append(abs(priceWithLogarithm - analyticPrice)/analyticPrice)

    # price and error generating the process by simulating the logarithm

    standardEulerBlackScholes = StandardEulerDiscretization(numberOfSimulations, timeStep, finalTime, initialValue,
                                                            muFunction = lambda t, x : r*x , sigmaFunction = lambda t, x : sigma*x)

    processRealizationsStandard = standardEulerBlackScholes.getRealizationsAtGivenTime(maturity)

    payoffsWithStandard = payoffFunction(processRealizationsStandard)
    priceWithStandard = math.exp(-r*maturity) * np.mean(payoffsWithStandard)

    errorsWithStandard.append((priceWithStandard - analyticPrice) / analyticPrice)



print("Average error simulating the logarithm: ", np.mean(errorsWithLogarithm))
print("Average error using the standard discretization of the SDE: ", np.mean(errorsWithStandard))
