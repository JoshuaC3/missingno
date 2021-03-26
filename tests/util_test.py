"""
Utilities test module. Asserts that utility functions are correct.
"""

import unittest
import pandas as pd
import numpy as np

import sys; sys.path.append("../")
import missingno as msno

from missingno import fillingno as flno


class TestNullitySort(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'A': [0, np.nan, np.nan], 'B': [0, 0, np.nan]})

    def test_no_op(self):
        expected = self.df
        result = msno.nullity_sort(self.df, sort=None)

        assert result.equals(expected)

    def test_ascending_sort(self):
        result = msno.nullity_sort(self.df, sort='ascending')
        expected = self.df.iloc[[2, 1, 0]]
        assert result.equals(expected)

    def test_descending_sort(self):
        result = msno.nullity_sort(self.df, sort='descending')
        expected = self.df.iloc[[0, 1, 2]]
        assert result.equals(expected)


class TestNullityFilter(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'A': [0, np.nan, np.nan], 'B': [0, 0, np.nan], 'C': [0, 0, 0]})

    def test_no_op(self):
        assert self.df.equals(msno.nullity_filter(self.df))
        assert self.df.equals(msno.nullity_filter(self.df, filter='top'))
        assert self.df.equals(msno.nullity_filter(self.df, filter='bottom'))

    def test_percentile_cutoff_top_p(self):
        expected = self.df.loc[:, ['B', 'C']]
        result = msno.nullity_filter(self.df, p=0.6, filter='top')
        assert result.equals(expected)

    def test_percentile_cutoff_bottom_p(self):
        expected = self.df.loc[:, ['A']]
        result = msno.nullity_filter(self.df, p=0.6, filter='bottom')
        assert result.equals(expected)

    def test_percentile_cutoff_bottom_n(self):
        expected = self.df.loc[:, ['C']]
        result = msno.nullity_filter(self.df, n=1, filter='top')
        assert result.equals(expected)

    def test_percentile_cutoff_top_n(self):
        expected = self.df.loc[:, ['A']]
        result = msno.nullity_filter(self.df, n=1, filter='bottom')
        assert result.equals(expected)

    def test_combined_cutoff_top(self):
        expected = self.df.loc[:, ['C']]
        result = msno.nullity_filter(self.df, n=2, p=0.7, filter='top')
        assert result.equals(expected)

    def test_combined_cutoff_bottom(self):
        expected = self.df.loc[:, ['A']]
        result = msno.nullity_filter(self.df, n=2, p=0.4, filter='bottom')
        assert result.equals(expected)


class TestKeyColumnsMask(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'A': [0, np.nan, np.nan], 'B': [0, 0, np.nan], 'C': [0, 0, 0]})

    def test_no_keys(self):
        df_, removed, color_df = flno.key_columns_mask(self.df, return_dfs=True)
        assert self.df.equals(df_)
        assert removed.empty
        assert color_df.eq(0).all().all()

    def test_single_key(self):
        df_, removed, color_df = flno.key_columns_mask(self.df, key_cols='B', return_dfs=True)
        assert self.df.drop(2).equals(df_)
        assert removed.equals(pd.DataFrame([[np.nan, np.nan, 0]], columns=['A', 'B', 'C'], index=[2]))
        pd.testing.assert_frame_equal(
            color_df,
            pd.DataFrame({'A': [0, 0, 2], 'B': [0, 0, 3], 'C': [0, 0, 2]}),
            check_dtype=False,
        )

    def test_multi_key(self):
        df_, removed, color_df = flno.key_columns_mask(self.df, key_cols=['A', 'B'], return_dfs=True)
        assert self.df.drop([1, 2]).equals(df_)
        assert removed.equals(
            pd.DataFrame([[np.nan, 0, 0], [np.nan, np.nan, 0]], columns=['A', 'B', 'C'], index=[1, 2])
        )
        pd.testing.assert_frame_equal(
            color_df,
            pd.DataFrame({'A': [0, 3, 3], 'B': [0, 2, 3], 'C': [0, 2, 2]}),
            check_dtype=False,
        )

    def test_key_single_output(self):
        color_df = flno.key_columns_mask(self.df, key_cols=['A', 'B'])
        assert color_df.equals(pd.DataFrame({'A': [0, 3, 3], 'B': [0, 2, 3], 'C': [0, 2, 2]}))


class TestPsuedoemptyColumnsMask(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'A': [0, np.nan, np.nan], 'B': [0, 0, np.nan], 'C': [0, 0, 0]})
        self.df_full = pd.DataFrame({'A': [0, 1, 1], 'B': [0, 0, 2], 'C': [0, 0, 0]})

    def test_no_change(self):
        df_, removed, color_df = flno.psuedoempty_columns_mask(self.df_full, return_dfs=True)
        assert self.df_full.equals(df_)
        assert removed.empty
        assert color_df.eq(0).all().all()

    def test_over_half(self):
        df_, removed, color_df = flno.psuedoempty_columns_mask(self.df, col_thresh=0.5, return_dfs=True)
        assert self.df.drop('A', axis=1).equals(df_)
        assert removed.equals(self.df[['A']])
        pd.testing.assert_frame_equal(
            color_df,
            pd.DataFrame({'A': [2, 3, 3], 'B': [0, 0, 0], 'C': [0, 0, 0]}),
            check_dtype=False,
        )

    def test_key_single_output(self):
        color_df = flno.psuedoempty_columns_mask(self.df_full)
        assert color_df.eq(0).all().all()


class TestPsuedoemptyRowsMask(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'A': [0, np.nan, np.nan], 'B': [0, 0, np.nan], 'C': [0, 0, 0]})
        self.df_full = pd.DataFrame({'A': [0, 1, 1], 'B': [0, 0, 2], 'C': [0, 0, 0]})

    def test_no_change(self):
        df_, removed, color_df = flno.psuedoempty_rows_mask(self.df_full, return_dfs=True)
        assert self.df_full.equals(df_)
        assert removed.empty
        assert color_df.eq(0).all().all()

    def test_over_half(self):
        df_, removed, color_df = flno.psuedoempty_rows_mask(self.df, row_thresh=0.5, return_dfs=True)
        assert self.df.drop(2).equals(df_)
        assert removed.equals(self.df.loc[[2], :])
        pd.testing.assert_frame_equal(
            color_df,
            pd.DataFrame({'A': [0, 0, 3], 'B': [0, 0, 3], 'C': [0, 0, 2]}),
            check_dtype=False,
        )

    def test_key_single_output(self):
        color_df = flno.psuedoempty_rows_mask(self.df_full)
        assert color_df.eq(0).all().all()


class TestInterpolationMask(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'A': [0, np.nan, np.nan], 'B': [0, 0, np.nan], 'C': [0, 0, 0]})
        self.df_full = pd.DataFrame({'A': [0, 1, 1], 'B': [0, 0, 2], 'C': [0, 0, 0]})

    def test_no_change(self):
        color_df = flno.interpolation_mask(self.df_full)
        assert color_df.eq(0).all().all()

    def test_filled_with_ones(self):
        color_df = flno.interpolation_mask(self.df)
        pd.testing.assert_frame_equal(
            color_df,
            pd.DataFrame({'A': [0, 1, 1], 'B': [0, 0, 1], 'C': [0, 0, 0]}),
            check_dtype=False,
        )


class TestUpdateColorDF(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'A': [0, 0, 2], 'B': [0, 0, 2], 'C': [0, 0, 3]})
        self.df_zeros = pd.DataFrame({'A': [0, 0, 0], 'B': [0, 0, 0], 'C': [0, 0, 0]})
        self.df_twos = pd.DataFrame({'A': [2, 0, 2], 'B': [2, 0, 2], 'C': [2, 0, 2]})

    def test_no_change(self):
        color_df = flno.update_color_df(self.df, self.df_zeros)
        assert color_df.equals(self.df)

    def test_filled_with_ones(self):
        color_df = flno.update_color_df(self.df, self.df_twos)
        pd.testing.assert_frame_equal(
            color_df,
            pd.DataFrame({'A': [2, 0, 2], 'B': [2, 0, 2], 'C': [2, 0, 3]}),
            check_dtype=False,
        )
