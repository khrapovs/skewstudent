#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""Skewed Student distribution [1].

The probability density function is given by

.. math::

    f\left(x|\eta,\lambda\right)=\begin{cases}
    bc\left(1+\frac{1}{\eta-2}\left(\frac{a+bx}{1-\lambda}\right)^{2}\right)
        ^{-\left(\eta+1\right)/2}, & x<-a/b,\\
    bc\left(1+\frac{1}{\eta-2}\left(\frac{a+bx}{1+\lambda}\right)^{2}\right)
        ^{-\left(\eta+1\right)/2}, & x\geq-a/b,
    \end{cases}

where :math:`2<\eta<\infty`, and :math:`-1<\lambda<1`.
The constants :math:`a`, :math:`b`, and :math:`c` are given by

.. math::

    a=4\lambda c\frac{\eta-2}{\eta-1},\quad b^{2}=1+3\lambda^{2}-a^{2},

and

.. math::

    c=\frac{\Gamma\left(\frac{\eta+1}{2}\right)}{\sqrt{\pi\left(\eta-2\right)}
        \Gamma\left(\frac{\eta}{2}\right)}.

A random variable with this density has mean zero and unit variance.
The distribution becomes Student t distribution when :math:`\lambda=0`.

References
----------

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

    Attributes
    ----------
    nup : float
        Degrees of freedom. 2 < \eta < \infty
    lam : float
        Skewness. -1 < \lambda < 1

    Methods
    -------
    pdf
        Probability density function (PDF)
    cdf
        Cumulative density function (CDF)
    icdf
        Inverse cumulative density function (ICDF)
    rvs
        Random variates with mean zero and unit variance

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

    def __const_a(self):
        """Compute a constant.

        Returns
        -------
        a : float

        """
        return 4*self.lam*self.__const_c()*(self.nup-2)/(self.nup-1)

    def __const_b(self):
        """Compute b constant.

        Returns
        -------
        b : float

        """
        return (1 + 3*self.lam**2 - self.__const_a()**2)**.5

    def __const_c(self):
        """Compute c constant.

        Returns
        -------
        c : float

        """
        return gamma((self.nup+1)/2) \
            / ((np.pi*(self.nup-2))**.5*gamma(self.nup/2))

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
        c = self.__const_c()
        a = self.__const_a()
        b = self.__const_b()

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
        a = self.__const_a()
        b = self.__const_b()

        y1 = (b*arg+a)/(1-self.lam) * (self.nup/(self.nup-2))**.5
        y2 = (b*arg+a)/(1+self.lam) * (self.nup/(self.nup-2))**.5

        cdf = (arg < -a/b) * (1-self.lam) * t.cdf(y1, self.nup)
        cdf += (arg >= -a/b) * ((1-self.lam)/2 \
            + (1+self.lam) * (t.cdf(y2, self.nup)-.5))

        return cdf

    def icdf(self, arg):
        """Inverse cumulative density function (ICDF).

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

        a = self.__const_a()
        b = self.__const_b()

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
        return self.icdf(uniform.rvs(size=size))

    def plot_pdf(self, arg=np.linspace(-2, 2, 100)):
        """Plot probability density function.

        Parameters
        ----------
        arg : array
            Grid of point to evaluate PDF at

        """
        scale = (self.nup/(self.nup-2))**.5
        plt.plot(arg, t.pdf(arg, self.nup, scale=1/scale),
                 label='t distribution')
        plt.plot(arg, self.pdf(arg), label='skew-t distribution')
        plt.legend()
        plt.show()

    def plot_cdf(self, arg=np.linspace(-2, 2, 100)):
        """Plot cumulative density function.

        Parameters
        ----------
        arg : array
            Grid of point to evaluate CDF at

        """
        scale = (self.nup/(self.nup-2))**.5
        plt.plot(arg, t.cdf(arg, self.nup, scale=1/scale),
                 label='t distribution')
        plt.plot(arg, self.cdf(arg), label='skew-t distribution')
        plt.legend()
        plt.show()

    def plot_icdf(self, arg=np.linspace(-.99, .99, 100)):
        """Plot inverse cumulative density function.

        Parameters
        ----------
        arg : array
            Grid of point to evaluate ICDF at

        """
        scale = (self.nup/(self.nup-2))**.5
        plt.plot(arg, t.ppf(arg, self.nup, scale=1/scale),
                 label='t distribution')
        plt.plot(arg, self.icdf(arg), label='skew-t distribution')
        plt.legend()
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
