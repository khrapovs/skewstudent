#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Skewed Student distribution.

.. [1] Hansen, B. E. (1994). Autoregressive conditional density estimation.
    International Economic Review, 35(3), 705–730.

"""

from __future__ import print_function, division

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from scipy.special import gamma
from scipy.stats import t

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
        """Probability density function.

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
        """Cumulative density function.

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
        a = 4*self.lam*c* (self.nup-2) / (self.nup-1)
        b = (1 + 3*self.lam**2 - a**2)**.5

        y1 = (b*arg+a)/(1-self.lam) * (self.nup/(self.nup-2))**.5
        y2 = (b*arg+a)/(1+self.lam) * (self.nup/(self.nup-2))**.5

        cdf = (arg<-a/b) * (1-self.lam) * t.cdf(y1, self.nup)
        cdf += (arg>=-a/b) * ((1-self.lam)/2
            + (1+self.lam) * (t.cdf(y2, self.nup)-.5))

        return cdf

    def plot_pdf(self, arg=np.linspace(-2, 2, 100)):
        """Plot probability density function.

        """
        plt.plot(arg, self.pdf(arg))
        plt.show()

    def plot_cdf(self, arg=np.linspace(-2, 2, 100)):
        """Plot probability density function.

        """
        plt.plot(arg, self.cdf(arg))
        plt.show()


if __name__ == '__main__':

    sns.set_context('notebook')
    skewt = SkewStudent(nup=3, lam=-.5)
    skewt.plot_pdf()
    skewt.plot_cdf()
