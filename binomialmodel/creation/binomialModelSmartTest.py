"""
Here we test the simulation of the binomial model whenwe don't rely on pseudo random numbers generations, but at a given
time we just consider ALL the possible realizations of the process attributing a (analytic!) probability to every
realization. In particular, we print and plot the evolution of the discounted average of the process and of the probability
that the discounted value of a future realization of the process is bigger than the initial value.

@author: Andrea Mazzon
"""
from binomialModelSmart import BinomialModelSmart

initialValue = 3.0
decreaseIfDown = 0.5
increaseIfUp = 2
numberOfTimes = 150
interestRate = 0.0

myBinomialModelSmart = BinomialModelSmart(initialValue, decreaseIfDown, increaseIfUp, numberOfTimes, interestRate)

#prints..
myBinomialModelSmart.printEvolutionPercentagesOfGain()
myBinomialModelSmart.printEvolutionDiscountedAverage()

#..and plots
myBinomialModelSmart.plotEvolutionPercentagesOfGain()
myBinomialModelSmart.plotEvolutionDiscountedAverage()



