"""
Here we test the simulation of the binomial model with the pure Monte Carlo approach. We print a single path of the
process, i.e., for a given simulation/state of the world. We also print and plot the evolution of the discounted
average and of the probability that the discounted value of a future realization of the process is bigger than the
initial value.

@author: Andrea Mazzon
"""
from binomialModelMonteCarlo import BinomialModelMonteCarlo

initialValue = 3
decreaseIfDown = 0.9
increaseIfUp = 1.1
numberOfTimes = 150
numberOfSimulations = 100000
interestRate = 0.05

myBinomialModel = BinomialModelMonteCarlo(initialValue, decreaseIfDown, increaseIfUp,numberOfTimes, numberOfSimulations,
                                          interestRate, 1897)

# we print the 10-th path of the process
simulationNumber = 10;
myBinomialModel.printPath(simulationNumber)

# and we plot some paths
myBinomialModel.plotPaths(10, 5)

# discounted average of the process and percentage of gains:

# prints..
myBinomialModel.printEvolutionDiscountedAverage()
myBinomialModel.printEvolutionPercentagesOfGain()
myBinomialModel.printEvolutionMaximum()

# ..and plots
myBinomialModel.plotEvolutionDiscountedAverage()
myBinomialModel.plotEvolutionPercentagesOfGain()
myBinomialModel.plotEvolutionMaximum()