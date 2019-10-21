"""Unit test for the pdf"""



import unittest

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
import scipy

from splrand.pdf import ProbabilityDensityFunction


class TestPdf(unittest.TestCase):

    def test_gauss(self, mu=0., sigma=1., support=10., num_points=500):
        '''Unit test with a gaussian distribution
        '''
        from scipy.stats import norm
        x = np.linspace(-support * sigma + mu, support * sigma + mu, num_points)
        y = norm.pdf(x, mu, sigma)
        pdf = ProbabilityDensityFunction(x, y)
        plt.plot(x, y)

        plt.figure('cdf')
        plt.plot(x, pdf.cdf(x))
        plt.xlabel('x')
        plt.ylabel('cdf(x)')

        plt.figure('ppf')
        q = np.linspace(0., 1., 1000)
        plt.plot(q, pdf.ppf(q))
        plt.xlabel('q')
        plt.ylabel('ppf(q)')

        plt.figure('Sampling')
        rnd = pdf.rnd(10000000)
        ydata, edges, _ = plt.hist(rnd, bins=200)
        xdata = 0.5 * (edges[1:] + edges [:-1])
        print(ydata.shape, edges.shape, xdata.shape)

        def f(x, C, mu, sigma):
            return C * norm.pdf(x, mu, sigma)

        popt, pcov = curve_fit(f, xdata, ydata)
        print(popt)
        print(np.sqrt(pcov.diagonal()))
        _x = np.linspace(-10, 10, 500)
        _y = f(_x, *popt)
        plt.plot(_x, _y)

        mask = ydata > 0
        chi2 = sum(((ydata[mask] - f(xdata[mask], *popt)) / np.sqrt(ydata[mask]))**2.)

        nu = mask.sum() - 3
        sigma = np.sqrt(2 * nu)
        print(chi2, nu, sigma)




if __name__ == '__main__':
    '''x = np.linspace(0., 1., 101)
    y = 2. * x
    pdf = ProbabilityDensityFunction(x, y)
    a = np.array([0.2, 0.6])
    print(pdf(a))

    plt.figure('pdf')
    plt.plot(x, pdf(x))
    plt.xlabel('x')
    plt.ylabel('pdf(x)')

    plt.figure('cdf')
    plt.plot(x, pdf.cdf(x))
    plt.xlabel('x')
    plt.ylabel('cdf(x)')

    plt.figure('ppf')
    q = np.linspace(0., 1., 250)
    plt.plot(q, pdf.ppf(q))
    plt.xlabel('q')
    plt.ylabel('ppf(q)')

    plt.figure('Sampling')
    rnd = pdf.rnd(1000000)
    plt.hist(rnd, bins=200)


    plt.show()'''

    unittest.main()
