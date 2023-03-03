"""
It tests the pricing of a Knock-out (down-and-out) option, by using the Monte-Carlo simulation of the underlying process

@author: Andrea Mazzon
"""

import time
import numpy as np

from processSimulation.eulerDiscretizationForBlackScholesWithLogarithm import EulerDiscretizationForBlackScholesWithLogarithm
from processSimulation.knockOutOption import KnockOutOption
from analyticformulas.analyticFormulas import blackScholesDownAndOut



numberOfSimulations = 10000
seed = 20

timeStep = 0.1
finalTime = 3
maturity = finalTime

initialValue = 2
r = 0.0
sigma = 0.5 


strike = initialValue
lowerBarrier = 1.1

analyticPrice = blackScholesDownAndOut(initialValue, r, sigma, maturity, strike, lowerBarrier)

print("The analytic price is ", analyticPrice)

#Monte-Carlo

payoffFunction = lambda x : np.maximum(x-strike,0)



timeMCInit = time.time() 

eulerBlackScholes= EulerDiscretizationForBlackScholesWithLogarithm(numberOfSimulations, timeStep, finalTime,
                   initialValue, r, sigma)

processRealizations = eulerBlackScholes.getRealizations()

knockOutOption = KnockOutOption(payoffFunction, maturity, lowerBarrier)

priceMonteCarlo = knockOutOption.getPrice(processRealizations)

timeNeededMC = time.time()  - timeMCInit

print("The Monte-Carlo price is ", priceMonteCarlo)
print("The Monte-Carlo relative error is ", abs((priceMonteCarlo -analyticPrice)/analyticPrice))

print()

print("The time needed with Monte-Carlo is ", timeNeededMC)

