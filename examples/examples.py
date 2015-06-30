#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Examples.

"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from scipy.optimize import minimize

from skewstudent import SkewStudent


if __name__ == '__main__':

    eta, lam = 5, -.5
    param = [eta, lam]

    sns.set_context('paper')
    skewt = SkewStudent(eta=eta, lam=lam)

    data = skewt.rvs(2000)

    sns.kdeplot(data)
    plt.show()

    bounds = [(2.01, 300), (-1, 1)]
    res = minimize(skewt.loglikelihood, [10, 0], args=(data,), method='SLSQP',
                   bounds=bounds)
    print(res)