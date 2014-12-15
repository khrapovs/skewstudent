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

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"


class SkewStudent(object):

    """Skewed Student distribution class.

    """

    def __init__(self, nup=10, lam=-.1):
        """Initialize the class.

        Parameters
        ----------
        nup : float
            Degrees of freedom
        lam : float
            Skewness

        """
        self.nup = nup
        self.lam = lam

    def pdf(self, arg):
        """Probability density function.

        """
        c = gamma((self.nup+1)/2)/((np.pi*(self.nup-2))**.5*gamma(self.nup/2))
        a = 4*self.lam*c*(self.nup-2)/(self.nup-1)
        b = (1 + 3*self.lam**2 - a**2)**.5

        pdf1 = b*c*(1 + 1/(self.nup-2)*((b*arg+a)/(1-self.lam))**2) \
            **(-(self.nup+1)/2)
        pdf2 = b*c*(1 + 1/(self.nup-2)*((b*arg+a)/(1+self.lam))**2) \
            **(-(self.nup+1)/2)

        return pdf1 * (arg < -a/b) + pdf2 * (arg >= -a/b)

    def plot_density(self, arg=np.linspace(-2, 2, 100)):
        """Plot probability density function.

        """
        plt.plot(arg, self.pdf(arg))
        plt.show()


if __name__ == '__main__':

    sns.set_context('notebook')
    skewt = SkewStudent()
    skewt.plot_density()
