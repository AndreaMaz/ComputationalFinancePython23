"""
Here we test the comparison of the average errors we get with standard Monte-Carlo
and Antithetic Variables when valuating an European call option. 

In particular, we plot the two average errors for an incresing number of
simulations.

@author: Andrea Mazzon
"""

from compareStandardMCWithAV import compare
import time

import matplotlib.pyplot as plt


#process parameters
initialValue = 100
r = 0.05
sigma = 0.5

#option parameters
T = 3
strike = initialValue


#we want to test the two methods for different numbers of simulations
numbersOfSimulations = [10**k for k in range(3,6)] #[1000, 10000, 100000]

averagePercentageErrorsWithStandardMC = []
averagePercentageErrorsWithAV = []

startingTime = time.time()

for numberOfSimulations in numbersOfSimulations:
    #the function compare returns a 2-uple. The first value is the average percentage error of the standard Monte-Carlo method,
    #the second one the one with Antithetic variables
    averagePercentageErrorWithStandardMC, averagePercentageErrorWithAV = compare(numberOfSimulations,initialValue, sigma, T, strike, r)
    averagePercentageErrorsWithStandardMC.append(averagePercentageErrorWithStandardMC)
    averagePercentageErrorsWithAV.append(averagePercentageErrorWithAV)

finalTime = time.time()

elapsedTime = finalTime - startingTime

print("The elapsed time is ", elapsedTime)

plt.plot(numbersOfSimulations,averagePercentageErrorsWithStandardMC, 'bo')
plt.plot(numbersOfSimulations,averagePercentageErrorsWithAV, 'ro')
plt.xlabel('Number of simulations')
plt.ylabel('Average percentage error')
plt.title('Average percentage errors for a call option, with standard M-C and Antithetic Variables')
plt.legend(['Standard Monte-Carlo', 'Antithetic Variables'])
plt.show()

