"""
Visualization test module. Asserts that visualization functions work properly.
"""

import unittest
import pandas as pd
import numpy as np
import pytest

import sys; sys.path.append("../")
from missingno import fillingno as flno
import matplotlib.pyplot as plt


class TestMatrix(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.simple_df = pd.DataFrame((np.random.random((20, 10)) > 0.5), columns=range(0, 10)).replace(False, np.nan)
        np.random.seed(42)
        self.freq_df = (
            pd.DataFrame((np.random.random(1000).reshape((50, 20)) > 0.5))
                .replace(False, np.nan)
                .set_index(pd.period_range('1/1/2011', '2/1/2015', freq='M'))
        )
        np.random.seed(42)
        self.large_df = pd.DataFrame((np.random.random((250, 60)) > 0.5)).replace(False, np.nan)

    @pytest.mark.mpl_image_compare
    def test_simple_matrix(self):
        flno.matrix(self.simple_df)
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_no_sparkline_matrix(self):
        flno.matrix(self.simple_df, sparkline=False)
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_width_ratios_matrix(self):
        flno.matrix(self.simple_df, width_ratios=(30, 1))
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_color_matrix(self):
        flno.matrix(self.simple_df, colors=[(105, 100, 35), (0, 50, 192), (255, 100, 0), (225, 50, 25)])
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_fontsize_matrix(self):
        flno.matrix(self.simple_df, fontsize=8)
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_freq_matrix(self):
        flno.matrix(self.freq_df, freq='BQ')
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_large_matrix(self):
        flno.matrix(self.large_df)
        return plt.gcf()
