"""
@author: Andrea Mazzon
"""
import numpy as np
import math

class GenerateBSReturnsWithArrays:
    """
    In this class we generate N realizations of the returns of a log-normal process
    dX_t = r X_t dt + sigma X_t dW_t, 0 \le t \le T,
    over the sub-intervals [t_j, t_{j+1]] of a given length, where (t_j)_{j=1,..,n} forms an equi-spaced partition of
    the interval [0, T].
    
    The returns are computed as
    X_{t_{j+1}}/X_{t_{j}} - 1 = exp((r- 0.5 sigma^2) dt + sigma dt^0.5 Z) - 1,
    where Z is a standard normal random variable and dt is the (constant) length of the intervals. They are stored in a
    matrix represented by a numpy array.
    
    We proceed in two different ways:
        - we generate the realizations of Z directly;
        - we first generate the half realizations of Z, and then set the others equal to the opposite of the already
        generated ones
        
    The second way accounts for the Antithetic Variables variance reduction method.

    Here we work with arrays.
    
    
    Attributes
    ----------
    numberOfSimulations : int
        the number of simulated paths of the returns
    numberOfIntervals : int
        the number of time intervals in the partition
    finalTime : float
        the final time of the process X
    sigma : float
        the log-volatility
    r : float
        the interest rate. 

    Methods
    -------
    generateReturns(self):
        It generates a returns a number N = self.numberOfSimulations of paths of the returns
        of the log-normal process of the time intervals
    generateReturnsAntitheticVariables(self):
        It generates a returns a number N = self.numberOfSimulations of paths of the returns
        of the log-normal process of the time intervals, via Antithetic Variables
    """
    
     #Python specific syntax for the constructor
    def __init__(self, numberOfSimulations, numberOfIntervals, finalTime, sigma,
                 r = 0):#r = 0 if not specified
        """    
        Parameters
        ----------
        numberOfSimulations : int
           the number of simulated paths of the returns
        numberOfIntervals : int
            the number of time intervals in the partition
        finalTime : float
            the final time of the process X
        sigma : float
            the log-volatility
        r : float
            the interest rate. Default = 0
        """
        self.numberOfSimulations = numberOfSimulations
        self.numberOfIntervals = numberOfIntervals
        self.finalTime = finalTime
        self.sigma = sigma
        self.r = r
        
        
    def generateReturns(self):
        """
        It generates a returns a number N = self.numberOfSimulations of paths of the returns of the log-normal process
        of the time intervals

        Returns
        -------
        blackScholesRealizations : list
            a matrix representing the returns of the process. Row i represents the returns for the i-th simulation

        """
        
        lenghthOfIntervals = self.finalTime / self.numberOfIntervals 
                
        #we don't want to compute this every time.
        firstPart = math.exp((self.r - 0.5 * self.sigma**2) * lenghthOfIntervals)
        
        standardNormalRealizations = np.random.standard_normal(size=(self.numberOfSimulations,self.numberOfIntervals))

        #try to use math.exp: what does it happen? why?
        blackScholesReturns = firstPart * np.exp(self.sigma * math.sqrt(lenghthOfIntervals) * standardNormalRealizations)
            
        return blackScholesReturns
       
        
    def generateReturnsAntitheticVariables(self):
        """
        It generates and returns a number N = self.numberOfSimulations of paths of the returns of the log-normal process
        of the time intervals, via Antithetic Variables
        
        Returns
        -------
        blackScholesRealizations : list
            a matrix representing the returns of the process. Row i represents the returns for the i-th simulation
        """
        
        halfSimulations = math.ceil(self.numberOfSimulations/2)
        lenghthOfIntervals = self.finalTime / self.numberOfIntervals 
                
        #we don't want to compute this every time.
        firstPart = math.exp((self.r - 0.5 * self.sigma**2) * lenghthOfIntervals)
        standardNormalRealizations = np.random.standard_normal((halfSimulations,self.numberOfIntervals))

        #try to use math.exp: what does it happen? why?
        firstBlackScholesReturns = firstPart * np.exp(self.sigma * math.sqrt(lenghthOfIntervals) * standardNormalRealizations)
        secondBlackScholesReturns = firstPart * np.exp(self.sigma * math.sqrt(lenghthOfIntervals) * (-standardNormalRealizations))

        blackScholesReturns = np.concatenate((firstBlackScholesReturns, secondBlackScholesReturns))
               
        return blackScholesReturns