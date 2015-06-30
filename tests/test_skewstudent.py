#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Testing suite for SkewStudent class.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
from scipy.stats import t

from skewstudent import SkewStudent

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"


class SkewStudentTestCase(ut.TestCase):

    """Test SkewStudent distribution class."""

    def test_init(self):
        """Test __init__."""

        skewt = SkewStudent()

        self.assertIsInstance(skewt.eta, float)
        self.assertIsInstance(skewt.lam, float)

        eta, lam = 5., -.2
        skewt = SkewStudent(eta=eta, lam=lam)

        self.assertEqual(skewt.eta, eta)
        self.assertEqual(skewt.lam, lam)

    def test_pdf(self):
        """Test pdf method."""

        skewt = SkewStudent()

        num = 50
        arg = np.linspace(-1, 1, num)
        pdf = skewt.pdf(arg)

        self.assertEqual(pdf.shape[0], num)
        self.assertIsInstance(skewt.pdf(0), float)

    def test_cdf(self):
        """Test cdf method."""

        skewt = SkewStudent()

        num = 50
        arg = np.linspace(-1, 1, num)
        cdf = skewt.cdf(arg)

        self.assertEqual(cdf.shape[0], num)
        self.assertIsInstance(skewt.cdf(0), float)

    def test_ppf(self):
        """Test ppf method."""

        skewt = SkewStudent()

        num = 50
        arg = np.linspace(.01, .99, num)
        ppf = skewt.ppf(arg)

        self.assertEqual(ppf.shape[0], num)
        self.assertIsInstance(skewt.ppf(.5), float)

    def test_rvs(self):
        """Test ppf method."""

        skewt = SkewStudent()

        rvs = skewt.rvs()

        self.assertIsInstance(rvs, float)

        size = 2
        rvs = skewt.rvs(size=size)

        self.assertIsInstance(rvs, np.ndarray)
        self.assertEqual(rvs.shape, (size, ))

        size = (2, 3)
        rvs = skewt.rvs(size=size)

        self.assertIsInstance(rvs, np.ndarray)
        self.assertEqual(rvs.shape, size)

    def test_compare_with_t(self):
        """Compare with standard t distribution."""

        eta = 5
        skewt = SkewStudent(eta=eta, lam=0)
        scale = 1/(eta/(eta-2))**.5
        standt = t(eta, scale=scale)
        arg = np.linspace(-2, 2, 100)

        np.testing.assert_array_almost_equal(skewt.pdf(arg), standt.pdf(arg))
        np.testing.assert_array_almost_equal(skewt.cdf(arg), standt.cdf(arg))

        arg = np.linspace(.01, .99, 100)

        np.testing.assert_array_almost_equal(skewt.ppf(arg), standt.ppf(arg))


if __name__ == '__main__':
    ut.main()
