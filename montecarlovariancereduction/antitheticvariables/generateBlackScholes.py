"""
@author: Andrea Mazzon
"""
import numpy as np
import math

from numpy import vectorize


class GenerateBlackScholes:
    """
    In this class we generate N realizations of a log-normal process
    dX_t = r X_t dt + sigma X_t dW_t
    at time T>0.
    
    We do it by writing
    X_T = X_0 exp((r- 0.5 sigma^2) T + sigma T^0.5 Z),
    where Z is a standard normal random variable.
    
    We proceed in two different ways:
        - we generate the N realizations of Z directly;
        - we first generate N/2 realizations of Z, and then set Z(j+N/2) = - Z(j).
        
    The second way accounts for the Antithetic Variables variance reduction method.
    
    
    Attributes
    ----------
    numberOfSimulations : int
        the number of simulated values of the process at maturity
    T : float
        the maturity of the option
    initialValue : float
        the initial value of the process
    sigma : float
        the standard deviation
    r : float
        the interest rate. Default = 0  
    randomNumberGenerator : np.random.RandomState
        Object which uses a Mersenne Twister pseudo-random number generator in order to provide samples of realizations
        from several probability distributions Default = None

    Methods
    -------
    generateRealizations(self):
        It generates a number N = self.numberOfSimulations of realizations of the log-normal process at time T,
        and returns the realizations as a list
    generateRealizationsAntitheticVariables(self):
        It generates a number N = self.numberOfSimulations of realizations of the log-normal process at time T,
        using Antithetic Variables, and returns the realizations as a list
    """
    
     #Python specific syntax for the constructor
    def __init__(self, numberOfSimulations, T, initialValue, sigma, r = 0,#r = 0 if not specified
                 seed = None):#no seed if not specified
        """    
        Parameters
        ----------
        numberOfSimulations : int
           the number of simulated values of the process at maturity
        T : float
            the maturity of the option
        initialValue : float
            the initial value of the process
        sigma : float
            the standard deviation
        r : float
            the interest rate. Default = 0
        seed : int
            the seed to generate the sequence both with standard Monte Carlo and Antithetic variables. Default = None
        """
        self.numberOfSimulations = numberOfSimulations
        self.T = T
        self.initialValue = initialValue
        self.sigma = sigma
        self.r = r

        # We now construct an object of type RandomState, which is a class of the package numpy.random. RandomState
        # uses a Mersenne Twister pseudo-random number generator in order to provide samples of realizations from
        # several probability distributions. It has a constructor with (only) parameter seed whose default value is None,
        # that is, no value(like null in Java).
        # Note now that if in the call of the constructor of our class we specify no seed at all, seed will have value
        # None. In this case, randomNumberGenerator gets constructed by calling np.random.RandomState().
        self.randomNumberGenerator = np.random.RandomState(seed)

        
    def generateRealizations(self):
        """
        It generates a number N = self.numberOfSimulations of realizations of the log-normal process at time T,
        and returns the realizations as a list.
        
        In particular, it does it by generating N values of standard normal random variables Z(j), j = 1, ..., N,
        and computing for every j
        X_T(j) = X_0 exp((r- 0.5 sigma^2) T + sigma T^0.5 Z(j))= X_0 exp((r- 0.5 sigma^2) T))exp(sigma T^0.5 Z(j))
        Returns
        -------
        blackScholesRealizations : list
            a list representing the realizations of the process
        """
                    
        # Note the way to get a given number of realizations of a standard normal random variable, as an array.
        # Also note that in order to access the specific field of the the class, we have to refer to it with "self.".
        # Same things for methods
        standardNormalRealizations = self.randomNumberGenerator.standard_normal(self.numberOfSimulations)

        #we don't want to compute this every time.
        firstPart = self.initialValue * math.exp((self.r - 0.5 * self.sigma**2) * self.T)

        BSFunction = lambda x : firstPart * math.exp(self.sigma * math.sqrt(self.T) * x)

        #or:
        #def BSFunction(x):
        #    return firstPart * math.exp(self.sigma * math.sqrt(self.T) * x)


        #Look at this peculiar Python for loop: this is equivalent to write
        #blackScholesRealizations = []
        #for k in range (standardNormalRealizations.length)
        #    blackScholesRealizations[k] = BSFunction(standardNormalRealizations[k])
        #The part (fox x in standardNormalRealizations) is similar to the Java foreach loop.
        #Such a loop returns a list
        blackScholesRealizations = [BSFunction(x) for x in standardNormalRealizations]

        #alternative way if you want blackScholesRealizations to be an array:
        #blackScholesRealizations = np.array([BSFunction(x) for x in standardNormalRealizations])

        #Alternative: vectorization is used to make the code faster, without using a loop. In this case, we would return an array
        #vectorizedBS = vectorize(BSFunction)

        #blackScholesRealizations = vectorizedBS(standardNormalRealizations)

        return blackScholesRealizations
       
        
    def generateRealizationsAntitheticVariables(self):
        """
        It generates a number N = self.numberOfSimulations of realizations of the log-normal process at time self.T,
        using Antithetic variables, and returns the realizations as a list.
        
        In particular, it does it by first generating N values of standard normal random variables
        Z(j), j = 1, ..., N/2, Z(n/2+j)=-Z(j), j = 1, ..., N/2
        and computing for every j
        X_T(j) = X_0 exp((r- 0.5 sigma^2) T + sigma T^0.5 Z(j)).
         
        If N is odd, N/2 is defined as the smallest integer >= N/2

        Returns
        -------
        blackScholesRealizations : list
            a list representing the realizations of the process

        """
        
        
        #math.ceil(x) returns the smallest integer >= x
        standardNormalRealizations = np.random.standard_normal(math.ceil(self.numberOfSimulations/2))
        
        #we don't want to compute this every time.
        firstPart = self.initialValue * math.exp((self.r - 0.5 * self.sigma**2) * self.T)

        BSFunction = lambda x: firstPart * math.exp(self.sigma * math.sqrt(self.T) * x)

        #note the use of the concatenation operator "+" between Python lists
        blackScholesRealizations = [BSFunction(x) for x in standardNormalRealizations] + \
                                 [BSFunction(-x) for x in standardNormalRealizations]

        # alternatively, to get an array:
        #blackScholesRealizations = np.array([BSFunction(x) for x in standardNormalRealizations] + \
        #                          [BSFunction(-x) for x in standardNormalRealizations])

        #or
        #vectorizedBS = vectorize(BSFunction)

        #blackScholesRealizations = vectorizedBS(np.concatenate((standardNormalRealizations,-standardNormalRealizations)) )

               
        return blackScholesRealizations