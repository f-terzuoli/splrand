"""Core logic for the pdf definition.
"""


import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


class ProbabilityDensityFunction(InterpolatedUnivariateSpline):

    """Class describing a probability density function.
    """

    def __init__(self, x, y):
        """Constructor.
        """
        InterpolatedUnivariateSpline.__init__(self, x, y)
        ycdf = np.array([self.integral(x[0], xcdf) for xcdf in x])
        self.cdf = InterpolatedUnivariateSpline(x, ycdf)
        """Need to make sure that the vector I am passingto the ppf spline as
        the x values has no duplicates---and need to filter the y accordingly."""
        xppf, ippf = np.unique(ycdf, return_index=True)
        yppf = x[ippf]

        self.ppf = InterpolatedUnivariateSpline(xppf, yppf)

    def prob(self, x1, x2):
        """Return the probability for the random variable to be included
        between x1 and x2.
        """
        return self.cdf(x2) - self.cdf(x1)

    def rnd(self, size=1000):
        """Return an array of random values from the pdf.
        """
        return self.ppf(np.random.uniform(size=size))
