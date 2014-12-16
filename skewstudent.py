#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Skewed Student distribution.

.. [1] Hansen, B. E. (1994). Autoregressive conditional density estimation.
    *International Economic Review*, 35(3), 705–730.

"""

from __future__ import print_function, division

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from scipy.special import gamma
from scipy.stats import t, uniform

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"


class SkewStudent(object):

    """Skewed Student distribution class.

    """

    def __init__(self, nup=10., lam=-.1):
        """Initialize the class.

        Parameters
        ----------
        nup : float
            Degrees of freedom. 2 < \eta < \infty
        lam : float
            Skewness. -1 < \lambda < 1

        """
        self.nup = nup
        self.lam = lam

    def pdf(self, arg):
        """Probability density function (PDF).

        Parameters
        ----------
        arg : array
            Grid of point to evaluate PDF at

        Returns
        -------
        pdf : array
            PDF values. Same shape as the input.

        """
        c = gamma((self.nup+1)/2)/((np.pi*(self.nup-2))**.5*gamma(self.nup/2))
        a = 4*self.lam*c*(self.nup-2)/(self.nup-1)
        b = (1 + 3*self.lam**2 - a**2)**.5

        pdf1 = b*c*(1 + 1/(self.nup-2)*((b*arg+a)/(1-self.lam))**2) \
            **(-(self.nup+1)/2)
        pdf2 = b*c*(1 + 1/(self.nup-2)*((b*arg+a)/(1+self.lam))**2) \
            **(-(self.nup+1)/2)

        return pdf1 * (arg < -a/b) + pdf2 * (arg >= -a/b)

    def cdf(self, arg):
        """Cumulative density function (CDF).

        Parameters
        ----------
        arg : array
            Grid of point to evaluate CDF at

        Returns
        -------
        cdf : array
            CDF values. Same shape as the input.

        """
        c = gamma((self.nup+1)/2) / \
            ((np.pi*(self.nup-2))**.5*gamma(self.nup/2))
        a = 4*self.lam*c * (self.nup-2) / (self.nup-1)
        b = (1 + 3*self.lam**2 - a**2)**.5

        y1 = (b*arg+a)/(1-self.lam) * (self.nup/(self.nup-2))**.5
        y2 = (b*arg+a)/(1+self.lam) * (self.nup/(self.nup-2))**.5

        cdf = (arg < -a/b) * (1-self.lam) * t.cdf(y1, self.nup)
        cdf += (arg >= -a/b) * ((1-self.lam)/2 \
            + (1+self.lam) * (t.cdf(y2, self.nup)-.5))

        return cdf

    def icdf(self, arg):
        """Inverse cumulative density function.

        Parameters
        ----------
        arg : array
            Grid of point to evaluate ICDF at. Must belong to (0, 1)

        Returns
        -------
        icdf : array
            ICDF values. Same shape as the input.

        """
        arg = np.atleast_1d(arg)

        c = gamma((self.nup+1)/2) / \
            ((np.pi*(self.nup-2))**.5 * gamma(self.nup/2))
        a = 4*self.lam*c * (self.nup-2) / (self.nup-1)
        b = (1 + 3*self.lam**2 - a**2)**.5

        f1 = arg < (1-self.lam)/2
        f2 = arg >= (1-self.lam)/2

        icdf1 = (1-self.lam)/b * ((self.nup-2)/self.nup)**.5 \
            * t.ppf(arg[f1]/(1-self.lam), self.nup)-a/b
        icdf2 = (1+self.lam)/b * ((self.nup-2)/self.nup)**.5 \
            * t.ppf(.5+1/(1+self.lam)*(arg[f2]-(1-self.lam)/2), self.nup)-a/b
        icdf = -999.99*np.ones_like(arg)
        icdf[f1] = icdf1
        icdf[f2] = icdf2

        if icdf.shape == (1, ):
            return float(icdf)
        else:
            return icdf

    def rvs(self, size=1):
        """Random variates with mean zero and unit variance.

        Parameters
        ----------
        size : int or tuple
            Size of output array

        Returns
        -------
        rvs : array
            Array of random variates

        """
        arg = uniform.rvs(size=size)
        return self.icdf(arg)

    def plot_pdf(self, arg=np.linspace(-2, 2, 100)):
        """Plot probability density function.

        Parameters
        ----------
        arg : array
            Grid of point to evaluate PDF at

        """
        plt.plot(arg, self.pdf(arg))
        plt.show()

    def plot_cdf(self, arg=np.linspace(-2, 2, 100)):
        """Plot cumulative density function.

        Parameters
        ----------
        arg : array
            Grid of point to evaluate CDF at

        """
        plt.plot(arg, self.cdf(arg))
        plt.show()

    def plot_icdf(self, arg=np.linspace(-.99, .99, 100)):
        """Plot inverse cumulative density function.

        Parameters
        ----------
        arg : array
            Grid of point to evaluate ICDF at

        """
        plt.plot(arg, self.icdf(arg))
        plt.show()

    def plot_rvspdf(self, arg=np.linspace(-2, 2, 100), size=1000):
        """Plot kernel density estimate of a random sample.

        Parameters
        ----------
        arg : array
            Grid of point to evaluate ICDF at. Must belong to (0, 1)

        """
        rvs = self.rvs(size=size)
        xrange = [arg.min(), arg.max()]
        sns.kdeplot(rvs, clip=xrange, label='kernel')
        plt.plot(arg, self.pdf(arg), label='true pdf')
        plt.xlim(xrange)
        plt.legend()
        plt.show()


if __name__ == '__main__':

    sns.set_context('notebook')
    skewt = SkewStudent(nup=3, lam=-.5)
    skewt.plot_pdf()
    skewt.plot_cdf()
    skewt.plot_icdf()
    skewt.plot_rvspdf()
