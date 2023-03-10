"""
author: Andrea Mazzon
"""
import numpy as np 

class AmericanOption:
    """
    This class is designed to valuate the price of an American option written on a binomial model for a given, general payoff.
    
    It is also possible to get matrices representing the values of the option and the amount of money one would get by
    waiting and by exercising the option, respectively, and the exercise region.
    
    Attributes
    ----------
    underlyingModel : binomial.creation.BinomialModel
        the underlying binomialmodel   

    Methods
    -------
    getValueOption(payoffFunction, maturity)
        It returns the value at zero of the american option of given maturity for the given payoff function
   
    getAnalysisOption(payoffFunction, maturity)
        For the given maturity and payoff function, it returns:
        - a matrix with the values of the american option at every time
        - a matrix with the amount of money one would get if exercising
        - a matrix with the amount of money one would get if waiting
        - a matrix with 1 when it's convenient to exercise the option and 0 if it's convenient to wait
    """
    def __init__(self, underlyingProcess):
        """

        Parameters
        ----------
        underlyingModel : binomial.creation.BinomialModel
            the underlying binomialmodel 

        Returns
        -------
        None.

        """
        
        self.underlyingProcess = underlyingProcess
        
    
    def getValueOption(self, payoffFunction, maturity):
        """
        It returns the value of the american option of maturity for the given payoff function

        Parameters
        ----------
        payoffFunction : function 
            the function representing the payoff.
        maturity : int
            the maturity of the option.

        Returns
        -------
        float
            the value of the option.

        """
        #here it is done with lists, also to show you some features like zip. You can change it to arrays.
        binomialModel = self.underlyingProcess 
        
        q = binomialModel.riskNeutralProbabilityUp
        rho = binomialModel.interestRate
        
        #we proceed backwards: we start from the payoff 
        processRealizations = binomialModel.getRealizationsAtGivenTime(maturity)
        payoffRealizations = [payoffFunction(x) for x in processRealizations]
        
        #note that since here we are only interested to the price, we don't  define any matrix but simply store the
        # successive values of the option in a vector that will be updated at every iteration of the for loop
        
        #at the beginning, the value of the option is equal to the payoff
        valuesOption = payoffRealizations
        
        for timeIndexBackward in range(maturity - 1,-1, -1):

            processRealizations = binomialModel.getRealizationsAtGivenTime(timeIndexBackward)               
            #the money we get if we exercise the option
            optionPart = [payoffFunction(x) for x in processRealizations]
            
            #the money we get if we wait: 
            #V(j,k)=qV(j+1,k+1)+(1-q)V(j+1,k+1), where j is time and k the number of ups up to the current time
            valuationPart = [(q * x + (1 - q) * y)/(1+rho) for x,y in zip(valuesOption[:-1],  valuesOption[1:])]

            #for arrays:
            #valuationPart = (q * valuesOption[:-1] + (1 - q) * valuesOption[1:])/(1+rho)


            #and then we take the maximums: these are the current values of the option
            valuesOption = np.maximum(optionPart, valuationPart)
        
        return valuesOption[0]
    
        
    
    
    def getAnalysisOption(self, payoffFunction, maturity):
        """
        It performs an analysis of the american option option, returning  matrices representing the discounted values of
        the option, the discounted amount of money one would get by waiting and by exercising the option, respectively,
        and the exercise region.

        Parameters
        ----------
        payoffFunction : function 
            the function representing the payoff.
        maturity : int
            the maturity of the option.

        Returns
        -------
        valuesOption : array
            a triangular matrix with the discounted values of the american option at every time
        valuesExercise : array
            a triangular matrix with the discounted amount of money one would get if exercising the option
        valuesIfWait : array
            a triangular matrix with the discounted amount of money one would get if waiting
        exercise : array
            a triangular matrix with 1 when it's convenient to exercise the option and 0 if it's convenient to wait.

        """
        
        binomialModel = self.underlyingProcess 
        q = binomialModel.riskNeutralProbabilityUp
        rho = binomialModel.interestRate
        
        #Here we store everything in matrices, since we want to return all the values over time. Of course, if we only
        #want the price we should call the method above
        valuesExercise = np.full((maturity + 1,maturity + 1),np.nan) 
        valuesIfWait = np.full((maturity + 1,maturity + 1),np.nan) 
        valuesOption = np.full((maturity + 1,maturity + 1),np.nan) 
        exercise = np.full((maturity + 1,maturity + 1),np.nan) 
        
        #we proceed backwards. We start from looking at the payoffs
        processRealizations = binomialModel.getRealizationsAtGivenTime(maturity)
        payoffRealizations = [payoffFunction(x) for x in processRealizations]
        
        #all the values at maturity times are equal to the payoff
        valuesOption[maturity,:] = payoffRealizations
        valuesIfWait[maturity,:] = payoffRealizations
        valuesExercise[maturity,:] = payoffRealizations
        #and of course we exercise the option
        exercise[maturity,:] = np.full(maturity + 1, True)
        
        for timeIndexBackward in range(maturity - 1,-1, -1):

            processRealizations = binomialModel.getRealizationsAtGivenTime(timeIndexBackward)
            #the money we get if we exercise the option
            optionPart = [payoffFunction(x) for x in processRealizations]   
           
            #the money we get if we wait: 
            #V(j,k)=qV(j+1,k+1)+(1-q)V(j+1,k+1), where j is time and k the number
            #of ups up to time
            valuationPart = q/(1+rho) * valuesOption[timeIndexBackward + 1, 0:(timeIndexBackward + 1)] + \
                (1-q)/(1+rho) * valuesOption[timeIndexBackward + 1, 1:(timeIndexBackward + 2)]
                
     
            #and then we take the maximums: these are the current values of the option  
            
            valuesOption[timeIndexBackward, 0:timeIndexBackward + 1] = np.maximum(optionPart, valuationPart)
            
            valuesExercise[timeIndexBackward, 0:timeIndexBackward + 1] = optionPart
            
            valuesIfWait[timeIndexBackward, 0:timeIndexBackward + 1] = valuationPart
            
            #this identifies the exercise region
            exercise[timeIndexBackward, 0:timeIndexBackward + 1] = optionPart > valuationPart
                
        
        return valuesOption, valuesExercise, valuesIfWait, exercise
    
    