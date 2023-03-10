"""
@author: Andrea Mazzon
"""

import numpy as np

import math

from analyticformulas.analyticFormulas import blackScholesPriceCall
from cliquetOption import CliquetOption
from generateBSReturns import GenerateBSReturns




class ControlVariatesCliquetBS:
    """
    This class is designed to compute the payoff of a Cliquet option written on a Black-Scholes model, with control variates.
    
    In particular, the underlying of the option is the log-normal process
    dX_t = r X_t dt + sigma X_t dW_t, 0 \le t \le T,
    where T is the maturity of the option.
    
    The payoff of the Cliquet option with local floor localFloor, local cap localCap, global floor globalFloor, global
    cap globalCap and partition of the interval
    
    0=t_0 <= t_1 <=..<= t_N =T
    
    is
    
    min(max(R_1^*+R_2^*+..+R_N^*, globalFloor), globalCap),
    
    where 
    
    R_k^* = min(max(R_k, localFloor), localCap)
    
    with
    
    R_k = X_{t_k}/X_{t_{k-1}} - 1.
    
    The partition is supposed here to be equi-spaced.                              
    
    The returns of the log-normal process are computed as
    X_{t_{j+1}}/X_{t_{j}} = exp((r- 0.5 sigma^2) dt + sigma dt^0.5 Z),
    where Z is a standard normal random variable and dt is the (constant) length of the intervals.
    
    The Control variates method is performed using the analytic price of the option with payoff
    
    R_1^*+R_2^*+..+R_N^*,
    
    Attributes
    ----------
    numberOfSimulations : int
        the number of simulated trajectories of the returns
    maturity : float
        the maturity of the option. It is also the final time of the global time interval, i.e., the last time of the
        partition
    numberOfIntervals : int
        the number of time intervals in the partition  
    localFloor : float
        the floor for the single return in the option
    localCap : float
        the cap for the single return in the option
    globalFloor : float
        the floor for the sum of the truncated returns
    globalcap : float
        the floor for the sum of the truncated returns 
    sigma : float
        the log-volatility of the underlying
    r : float
        the interest rate. 
        

    Methods
    -------

    """
    
    def __init__(self, numberOfSimulations, maturity, numberOfIntervals, localFloor, localCap, globalFloor, globalCap, sigma, r):
        """
        Parameters
        ----------
        numberOfSimulations : int
            the number of simulated trajectories of the returns
        maturity : float
            the maturity of the option. It is also the final time of the global time interval, i.e., the last time of
            the partition
        numberOfIntervals : int
            the number of time intervals in the partition  
        localFloor : float
            the floor for the single return in the option
        localCap : float
            the cap for the single return in the option
        globalFloor : float
            the floor for the sum of the truncated returns
        globalcap : float
            the floor for the sum of the truncated returns 
        sigma : float
            the log-volatility of the underlying
        r : float
            the interest rate.     

        Returns
        -------
        None.

        """
        self.numberOfSimulations = numberOfSimulations
        
        self.maturity = maturity
        self.numberOfIntervals = numberOfIntervals
        
        self.localFloor = localFloor
        self.localCap = localCap
        
        self.globalFloor = globalFloor
        self.globalCap = globalCap
        
        self.sigma = sigma
        self.r = r


    def getAnalyticPriceOfNonTruncatedSum(self):
        """
        It returns the discounted analytic price of the derivative that pays the (non truncated) sum of the truncated
        returns at maturity.
        
        We exploit the fact that
        
        R_k^* = min(max(X_{t_k}/X_{t_{k-1}} - 1, F), C) = F + (Y_k -(F+1))^+ - (Y_k -(C+1))^+
              
        where 
        
        Y_k = R_k + 1 = X_k/X_{k-1}.
                               

        Returns
        -------
        analyticPrice : float
            the discounted analytic price of the non truncated sum of the truncated returns at maturity.

        """
        
        #we are considering an option on the return
        initialValue = 1
        
        maturityOfTheCallOptions = self.maturity/self.numberOfIntervals
        
        firstStrike = self.localFloor + 1
        secondStrike = self.localCap + 1

        localDiscountFactor = math.exp(-self.r * maturityOfTheCallOptions)

        firstCallPrice = blackScholesPriceCall(initialValue, self.r, self.sigma, maturityOfTheCallOptions, firstStrike)/localDiscountFactor

        secondCallPrice = blackScholesPriceCall(initialValue, self.r, self.sigma, maturityOfTheCallOptions, secondStrike)/localDiscountFactor

        #we repeat the same over all the time intervals, so we multiply by their number
        price = self.numberOfIntervals * (self.localFloor + firstCallPrice - secondCallPrice)
        
        #we now discount the price with respect to the maturity of the Cliquet option
        discountFactorCliquetOption = math.exp(- self.r * self.maturity)

        return discountFactorCliquetOption * price
    
    
    def getPriceViaControlVariates(self):  
        """
        It returns the discounted price of a Cliquet option using control variates.
        
        In particular, the Control variates method is performed using the analytic price of the
        option with payoff
    
        R_1^*+R_2^*+..+R_N^*,

        Returns
        -------
        float
            the discounted price of the option.

        """

        #just in order to deal with shorter expressions..

        numberOfSimulations = self.numberOfSimulations
        
        T = self.maturity
        numberOfIntervals = self.numberOfIntervals
        
        lF = self.localFloor
        lC = self.localCap
        
        gF = self.globalFloor
        gC = self.globalCap
        
        sigma = self.sigma
        r = self.r

        #first, we construct the object which represents the Cliquet option
        cliquetOption = CliquetOption(numberOfSimulations, T, lF, lC, gF, gC)

        #we then generate the Black-Scholes returns
        generator = GenerateBSReturns(numberOfSimulations, numberOfIntervals, T, sigma, r)
        
        returnsRealizations = generator.generateReturns()


        #first we get the Monte-Carlo price of the option..
        discountedPriceOfTheOptionMC =  cliquetOption.getDiscountedPriceOfTheOption(returnsRealizations, r)
        
        #..and then we want to get the analytic price and the Monte-Carlo price of the option in the case where there is
        #no truncation of the final sum:
    
        #first the Monte-Carlo price: see here two ways to represent infinity
        globalFloorForNonTruncatedSum = - np.inf
        globalCapForNonTruncatedSum = float('inf')
        
        cliquetOptionForNonTruncatedSum = CliquetOption(numberOfSimulations, T, lF, lC, globalFloorForNonTruncatedSum,
                                                        globalCapForNonTruncatedSum)
        
        discountedPriceNonTruncatedSumMC = \
            cliquetOptionForNonTruncatedSum.getDiscountedPriceOfTheOption(returnsRealizations, r)
        
        #and now the analytic value
        analyticPriceOfNonTruncatedSum = self.getAnalyticPriceOfNonTruncatedSum()
        
        #now we want to compute the optimal beta, see the script
        payoffsWhenTruncated = cliquetOption.getPayoffs(returnsRealizations)
        payoffsWhenNotTruncated = cliquetOptionForNonTruncatedSum.getPayoffs(returnsRealizations)  

        covarianceMatrix = np.cov(payoffsWhenTruncated, payoffsWhenNotTruncated)
        
        optimalBeta = covarianceMatrix[0,1]/covarianceMatrix[1,1]
        
        #and we return the price with control variates
        return discountedPriceOfTheOptionMC \
            - optimalBeta * (discountedPriceNonTruncatedSumMC - analyticPriceOfNonTruncatedSum)
