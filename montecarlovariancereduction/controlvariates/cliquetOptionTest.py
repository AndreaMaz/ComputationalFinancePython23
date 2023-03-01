"""
We test here the performances of three methods for the computation of the price of a Cliquet option, in terms of the
variance of the computed prices:
    - standard Monte-Carlo
    - Monte-Carlo with Antithetic variables
    - Monte-Carlo with Control variates.

We look at the variance since the analytic price is in general not known, so there is no benchmark to compute the error

@author: Andrea Mazzon
"""

import numpy as np
import time

from cliquetOption import CliquetOption
from cliquetOptionWithArrays import CliquetOptionWithArrays

from controlVariatesCliquetBS import ControlVariatesCliquetBS

from generateBSReturns import GenerateBSReturns
from generateBSReturnsWithArrays import GenerateBSReturnsWithArrays

# processParameters
from montecarlovariancereduction.controlvariates.fasterControlVariatesCliquetBS import FasterControlVariatesCliquetBS
from montecarlovariancereduction.controlvariates.fasterControlVariatesCliquetBSWithArrays import \
    FasterControlVariatesCliquetBSWithArrays

r = 0.2
sigma = 0.5

# option parameters
maturity = 4

numberOfTimeIntervals = 16

localFloor = -0.05
localCap = 0.3

globalFloor = 0
globalCap = numberOfTimeIntervals * 0.3

# Monte Carlo parameter

numberOfSimulations = 10000

# the objects to generate the returns
generator = GenerateBSReturns(numberOfSimulations, numberOfTimeIntervals, maturity, sigma, r)
generatorWithArrays = GenerateBSReturnsWithArrays(numberOfSimulations, numberOfTimeIntervals, maturity, sigma, r)


# we want to compute the price with the standard Cliquet option implementation..
cliquetOption = CliquetOption(numberOfSimulations, maturity, localFloor, localCap, globalFloor, globalCap)


# ..and then with the Cliquet option implementation that uses arrays
cliquetOptionWithArrays = CliquetOptionWithArrays(numberOfSimulations, maturity, localFloor, localCap, globalFloor, globalCap)


# ..with control variates..
cliquetOptionWithControlVariates = ControlVariatesCliquetBS(numberOfSimulations, maturity, numberOfTimeIntervals,
                             localFloor, localCap, globalFloor, globalCap, sigma, r)

# ..with the faster implementation of control variates..
fasterCliquetOptionWithControlVariates = \
    FasterControlVariatesCliquetBS(numberOfSimulations, maturity, numberOfTimeIntervals,
                                   localFloor, localCap, globalFloor, globalCap, sigma, r)

# ..and with faster implementation of control variates, using arrays
fasterCliquetOptionWithControlVariatesWithArrays = \
    FasterControlVariatesCliquetBSWithArrays(numberOfSimulations, maturity, numberOfTimeIntervals,
                                   localFloor, localCap, globalFloor, globalCap, sigma, r)

numberOfTests = 30

pricesStandard = []
pricesAV = []
pricesCV = []
pricesStandardWithArrays = []
pricesFasterCV = []
pricesFasterCVWithArrays = []

timesStandard = []
timesAV = []
timesCV = []
timesStandardWithArrays = []
timesFasterCV = []
timesFasterCVWithArrays = []

for k in range(numberOfTests):
    # first we do it via standard Monte-Carlo
    start = time.time()
    returnsRealizations = generator.generateReturns()
    priceStandardMC = cliquetOption.getDiscountedPriceOfTheOption(returnsRealizations, r)
    end = time.time()
    pricesStandard.append(priceStandardMC)
    timesStandard.append(end - start)

    # then via Monte-Carlo with Antithetic variables
    start = time.time()
    returnsRealizationsAV = generator.generateReturnsAntitheticVariables()
    priceAV = cliquetOption.getDiscountedPriceOfTheOption(returnsRealizationsAV, r)
    end = time.time()
    pricesAV.append(priceAV)
    timesAV.append(end - start)

    # then with control variates
    start = time.time()
    returnsRealizationsCV = generator.generateReturnsAntitheticVariables()
    priceCV = cliquetOptionWithControlVariates.getPriceViaControlVariates()
    end = time.time()
    pricesCV.append(priceCV)
    timesCV.append(end - start)

    # ..and then with control variates using arrays
    start = time.time()
    returnsRealizationsWithArrays = generatorWithArrays.generateReturns()
    priceStandardWithArrays = cliquetOptionWithArrays.getDiscountedPriceOfTheOption(returnsRealizationsWithArrays, r)
    end = time.time()
    pricesStandardWithArrays.append(priceStandardWithArrays)
    timesStandardWithArrays.append(end - start)

    # ..with the faster control variates
    start = time.time()
    priceFasterCV = fasterCliquetOptionWithControlVariates.getPriceViaControlVariates()
    end = time.time()
    pricesFasterCV.append(priceFasterCV)
    timesFasterCV.append(end - start)

    # ..and finally with the faster control variates using arrays
    start = time.time()
    priceFasterCVWithArrays = fasterCliquetOptionWithControlVariatesWithArrays.getPriceViaControlVariates()
    end = time.time()
    pricesFasterCVWithArrays.append(priceFasterCVWithArrays)
    timesFasterCVWithArrays.append(end - start)


print()
print("The variance of the prices using standard Monte-Carlo is ", np.var(pricesStandard))

print()
print("The variance of the prices using Antithetic variables is ", np.var(pricesAV))

print()
print("The variance of the prices using Control variates is ", np.var(pricesCV))

print()
print("The variance of the prices using standard Monte-Carlo with arrays is ", np.var(pricesStandardWithArrays))

print()
print("The variance of the prices using faster ocntrol variates is ", np.var(pricesFasterCV))

print()
print("The variance of the prices using the faster Control variates with arrays is ", np.var(pricesFasterCVWithArrays))

print()
print()
print("The average elapsed time using standard Monte-Carlo is ", np.mean(timesStandard))

print()
print("The average elapsed time using Antithetic variables is ", np.mean(timesAV))

print()
print("The average elapsed time using Control variates is ", np.mean(timesCV))

print()
print("The average elapsed time using standard Monte-Carlo with arrays is ", np.mean(timesStandardWithArrays))

print()
print("The average elapsed time using the faster Control variates is ", np.mean(timesFasterCV))

print()
print("The average elapsed time using the faster Control variates with arrays is ", np.mean(timesFasterCVWithArrays))
