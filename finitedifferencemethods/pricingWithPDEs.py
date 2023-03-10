"""
@author: Andrea Mazzon
"""

import abc
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import vectorize


class PricingWithPDEs(metaclass=abc.ABCMeta):
    """
    This class is devoted to numerically solve a general PDE 
    
        U_t + F[U_x,U_{xx},U_{xxx},..] = C(t),
        
    still not specifying the method, over the spatial domain of xmin <= x <= xmax that is discretized with a given step
    dx, and a time domain 0 <= t <= tmax that is discretized with a given time step dt.
    
    Boundary conditions given as attributes of the class are applied at both ends of the domain. An initial condition is
    applied at t = 0. This can be seen as the payoff of an option. In this case, time represents maturity.
    
    It is extended by classes representing the specific PDE and the specific method.
    
    Attributes
    ----------
    dx : float
        discretization step on the space
    dt : float
        discretization step on the time
    xmin : float
        left end of the space domain
    xmax : float
        right end of the space domain
    tmax : float
        right end of the time domain
    x : array
        the discretized space domain
    numberOfSpaceSteps : int
        the number of intervals of the space domain
    numberOfTimeSteps : int
        the number of intervals of the time domain
    payoff : function
        the initial condition. Called in this way because it corresponds to payoff of an option seeing time as maturity
    functionLeft : function
        the condition at the left end of the space domain
    functionRight : function
        the condition at the right end of the space domain
    currentTime : int
        the current time. The PDE is solved going forward in time. Here the current time is used to plot the solution
        dynamically and to compute the solution at the next time step in the derived classes.

    Methods
    -------
    getSolutionAtNextTime():
        It returns the solution at the next time step. It depends on the methods used. Abstract method in the parent class
    solveAndPlot():
        It solves the PDE and dynamically plots the solution at every time step of length 0.1. It does not store the
        solution in a matrix
    solveAndSave():
        It solves the PDE and store the solution as a matrix in the self.solution attribute of the class. It also returns it.
    getSolutionForGivenTimeAndValue(time, space):
        It returns the solution at given time (seen as time to maturity if we think about the evaluation of options) and
        given space
    """
    
    def __init__(self, dx, dt, xmin, xmax, tmax, payoff, functionLeft, functionRight):
        """
        Parameters
        ----------
        dx : float
            discretization step on the space
        dt : float
            discretization step on the time
        xmin : float
            left end of the space domain
        xmax : float
            right end of the space domain
        tmax : float
            right end of the time domain
        payoff : function
            the initial condition. Called in this way because it corresponds to payoff of an option seeing time as maturity
        functionLeft : function
            the condition at the left end of the space domain
        functionRight : function
            the condition at the right end of the space domain


        Returns
        -------
        None.

        """
        self.dx = dx # discretization step on the space
        self.dt = dt # discretization step on the time
        self.xmin = xmin
        self.xmax = xmax
        self.tmax = tmax
        
        #we create an equi-spaced space discretization
        self.x = np.arange(self.xmin, self.xmax+self.dx, self.dx) 
        self.numberOfSpaceSteps = math.ceil((self.xmax-self.xmin)/self.dx)
        self.numberOfTimeSteps = math.ceil(self.tmax/self.dt)

        #conditions
        self.payoff = payoff
        self.functionLeft = functionLeft
        self.functionRight = functionRight

        self.solution = None

    def __initializeU(self):
        #here we initialize the solution, u0 stores the initial condition
        payoffVectorized = vectorize(self.payoff)

        u0 = payoffVectorized(self.x)#x is an array: we can directly apply the vectorized payoff
        self.uCurrent = u0
        self.uPast = u0
    
    @abc.abstractmethod
    def getSolutionAtNextTime(self):
        """    
        It returns the solution at the next time step
        Parameters
        ----------
        None

        Returns
        -------
        The solution at the next time step

        """
        
    def solveAndPlot(self):
        """
        It solves the PDE and dynamically plots the solution at every time step of length 0.1. It does not store the solution in a matrix

        Returns
        -------
        None.

        """
        self.__initializeU()
        self.currentTime = 0
        timeToPlot = 0 # we want to plot at times 0, 0.1, 0.2,..
        for timeIndex in range(self.numberOfTimeSteps+2):
            #we get the new solution. The solution will be computed in the derived classes according to self.currentTime
            #and self.uPast
            self.uCurrent = self.getSolutionAtNextTime()
            #and update uPast: u will be the "past solution" at the next time step
            self.uPast = self.uCurrent
            
            #we plot the solution when currentTime is (close to) 0.1, 0.2, .
            # self.currentTime - self.dt <timeToPlot <= currentTime
            if self.currentTime - self.dt < timeToPlot and self.currentTime >= timeToPlot:
                plt.plot(self.x, self.uCurrent, 'bo-', label="Numeric solution")
                #we assume here that the solution is not bigger than the max x (generally true for options): then we set
                # x[-1] to be the max y axis
                plt.axis((self.xmin-0.12, self.xmax+0.12, 0, self.x[-1]))
                plt.grid(True)
                plt.xlabel("Underlying value")
                plt.ylabel("Price")
                plt.suptitle("Time = %1.3f" % timeToPlot)
                plt.pause(0.01)
                timeToPlot += 0.1
            self.currentTime += self.dt

        plt.show()


    def solveAndSave(self):
        """
        It solves the PDE and store the solution as a matrix in the self.solution
        attribute of the class. It also returns it.

        Returns
        -------
        array :
            The matrix representing the solution. Row k is the solution at time
            t_k = dt * k

        """
        self.__initializeU()
        self.currentTime = 0
        self.solution = np.zeros((self.numberOfTimeSteps+1,self.numberOfSpaceSteps+1))
        for i in range(self.numberOfTimeSteps+1):
            #we store the solution at past time
            self.solution[i] = self.uCurrent
            #we get the solution at current time. The solution will be computed in the
            #derived classes according to self.currentTime and self.uPast                    
            self.uCurrent = self.getSolutionAtNextTime()
            self.uPast = self.uCurrent
            self.currentTime += self.dt

    #self.solution[0]=u0

    #self.uCurrent=u1 (which is computed based on uPast (which is still u0) and currentTime (which is 0)
    #self.uPast = u1
    #self.currentTime = t_1

    # self.solution[1]=u1
    #self.uCurrent=u2 (which is computed based on uPast (which is now u1) and currentTime (which is now t_1)
    # self.uPast = u2
    #self.currentTime = t2

      
    def getSolutionForGivenTimeAndValue(self, time, space):
        """
        It returns the solution at given time and given space.

        Parameters
        ----------
        time : float
            the time: it represents maturiity for options
        space : float
            the space: it represents the underlying for options

        Returns
        -------
        float
            the solution at given time and space

        """
        #we generate the solution only once
        if self.solution is None:
           self.solveAndSave()
        #or:

        #we have to get the time and space indices
        timeIndexForTime = round(time/self.dt)#i such that t_i is closest to time
        spaceIndexForSpace = round((space - self.xmin)/self.dx)#j such that x_j is closest to space
        return self.solution[timeIndexForTime, spaceIndexForSpace]

