"""
This script is devoted to compare the gain we obtain when using Monte-Carlo
with Antithetic variables in the valuation of an European call option written
on a Black-Scholes model.
    
@author: Andrea Mazzon
"""

import numpy as np
from statistics import mean

from generateBlackScholes import GenerateBlackScholes
from simpleEuropeanOption import SimpleEuropeanOption
from analyticformulas.analyticFormulas import blackScholesPriceCall
        
    
def compare(numberOfSimulations, initialValue, sigma, T, strike, r = 0):
    """
    It returns the average percentage errors in the valuation of the call option we get
    using the standard Monte-Carlo method and the Monte-Carlo method with Antithetic Variables, respectively, over 500 tests.

    Parameters
    ----------
    numberOfSimulations : int
        the number of simulations of the the values of the process at maturity.
    initialValue : float
        the initial value of the process
    r : float
        the risk free rate
    sigma : float
        the log-volatility
    T : float
        the maturity of the option.
    strike : float
        the strike of the option.

    Returns
    -------
    averagePercentageErrorStandardMC : float
        the average percentage error we get by using the standard Monte-Carlo method over 500 tests.
    averagePercentageErrorAV : float
        the average percentage error that we get by using the Monte-Carlo method with Antithetic Variables over 500 tests
    """
        
    numberOfTests = 500
    
    
    #the two lists that will contain our average percentage errors for the different tests
    percentageErrorsStandardMonteCarlo = []
    percentageErrorsMonteCarloWithAV = []
    
    #our benchmark: the analytic price of the call option
    analyticPriceBS = blackScholesPriceCall(initialValue, r, sigma, T, strike)

    #look at how lambda functions are defined in Python 
    payoff = lambda x : max(x - initialValue, 0)#at the money option
    #alternatively:
    #def payoff(x):
    #    return max(x - initialValue, 0)
    
    #note how to construct an object of a class
    blackScholesGenerator = GenerateBlackScholes(numberOfSimulations, T, initialValue, sigma, r)

    #k=0,..,numberOfTests - 1
    for k in range(numberOfTests):

        #first, the valuation with the standard Monte-Carlo:
        realizationsOfTheProcessWithStandardMC = blackScholesGenerator.generateRealizations()
    
        priceCalculatorWithStandardMC = SimpleEuropeanOption(realizationsOfTheProcessWithStandardMC, r)
        priceStandardMC = priceCalculatorWithStandardMC.getPrice(payoff, T) 
        percentageErrorWithStandardMC = abs(priceStandardMC - analyticPriceBS)/analyticPriceBS*100
        #note how to append new values to a list
        percentageErrorsStandardMonteCarlo.append(percentageErrorWithStandardMC)
        
        #then, the one with Antithetic Variables:
        realizationsOfTheProcessWithAV = blackScholesGenerator.generateRealizationsAntitheticVariables()
        
        priceCalculatorWithAV = SimpleEuropeanOption(realizationsOfTheProcessWithAV, r)
        priceWithAV = priceCalculatorWithAV.getPrice(payoff, T) 
        percentageErrorWithAV= abs(priceWithAV - analyticPriceBS)/analyticPriceBS*100
        #note how to append new values to a list
        percentageErrorsMonteCarloWithAV.append(percentageErrorWithAV)

    #we get and return the respective average percentage errors: mean is imported from statistics
    averagePercentageErrorStandardMC = mean(percentageErrorsStandardMonteCarlo)
    #you can also use numpy
    averagePercentageErrorAV = np.mean(percentageErrorsMonteCarloWithAV)
    
    return averagePercentageErrorStandardMC, averagePercentageErrorAV
