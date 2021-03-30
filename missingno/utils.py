"""Utility functions for missingno."""

import numpy as np
import pandas as pd


def nullity_sort(df, sort=None, axis='columns'):
    """
    Sorts a DataFrame according to its nullity, in either ascending or descending order.

    :param df: The DataFrame object being sorted.
    :param sort: The sorting method: either "ascending", "descending", or None (default).
    :return: The nullity-sorted DataFrame.
    """
    if sort is None:
        return df
    elif sort not in ['ascending', 'descending']:
        raise ValueError('The "sort" parameter must be set to "ascending" or "descending".')

    if axis not in ['rows', 'columns']:
        raise ValueError('The "axis" parameter must be set to "rows" or "columns".')

    if axis == 'columns':
        if sort == 'ascending':
            return df.iloc[np.argsort(df.count(axis='columns').values), :]
        elif sort == 'descending':
            return df.iloc[np.flipud(np.argsort(df.count(axis='columns').values)), :]
    elif axis == 'rows':
        if sort == 'ascending':
            return df.iloc[:, np.argsort(df.count(axis='rows').values)]
        elif sort == 'descending':
            return df.iloc[:, np.flipud(np.argsort(df.count(axis='rows').values))]


def nullity_filter(df, filter=None, p=0, n=0):
    """
    Filters a DataFrame according to its nullity, using some combination of 'top' and 'bottom' numerical and
    percentage values. Percentages and numerical thresholds can be specified simultaneously: for example,
    to get a DataFrame with columns of at least 75% completeness but with no more than 5 columns, use
    `nullity_filter(df, filter='top', p=.75, n=5)`.

    :param df: The DataFrame whose columns are being filtered.
    :param filter: The orientation of the filter being applied to the DataFrame. One of, "top", "bottom",
    or None (default). The filter will simply return the DataFrame if you leave the filter argument unspecified or
    as None.
    :param p: A completeness ratio cut-off. If non-zero the filter will limit the DataFrame to columns with at least p
    completeness. Input should be in the range [0, 1].
    :param n: A numerical cut-off. If non-zero no more than this number of columns will be returned.
    :return: The nullity-filtered `DataFrame`.
    """
    if filter == 'top':
        if p:
            df = df.iloc[:, [c >= p for c in df.count(axis='rows').values / len(df)]]
        if n:
            df = df.iloc[:, np.sort(np.argsort(df.count(axis='rows').values)[-n:])]
    elif filter == 'bottom':
        if p:
            df = df.iloc[:, [c <= p for c in df.count(axis='rows').values / len(df)]]
        if n:
            df = df.iloc[:, np.sort(np.argsort(df.count(axis='rows').values)[:n])]
    return df


def key_columns_mask(df, key_cols=None, return_dfs=None):
    """
    Create a mask for columns with missing data in any key columns, typically the dependent variable.
    Fills the casual elements with value `3` and any affected values with `2`.

    :param df: The `DataFrame` being processed.
    :param key_cols: The key columns to remove null rows from.
    :param return_dfs: whether to return the transformed df_ and dropped rows.
    :return: color_df or df_, removed and color_df.
    """
    color_df = pd.DataFrame(np.zeros_like(df), index=df.index, columns=df.columns).astype(int)

    if isinstance(key_cols, str):
        key_cols = [key_cols]
    elif key_cols is None:
        #early exit
        if (return_dfs is None) or (not return_dfs):
            return color_df
        else:
            df_ = df.copy()
            #keep consistency on `removed`
            removed = pd.DataFrame(columns=df.columns).rename_axis(df.index.name)
            return df_, removed, color_df

    nonkey_cols = df.columns[~df.columns.isin(key_cols)]
    nonkey_color_mask = df.loc[:, key_cols].isna().any(axis=1)

    for col in key_cols:
        color_df.loc[df.loc[:, col].isna(), col] = 3
        color_df.loc[(nonkey_color_mask & df.loc[:, col].notna()), col] = 2

    color_df.loc[nonkey_color_mask, nonkey_cols] = 2

    if return_dfs is None or not return_dfs:
        return color_df
    else:
        df_ = df.copy()
        df_ = df.loc[~nonkey_color_mask, :]
        removed = df.loc[nonkey_color_mask, :]
        return df_, removed, color_df


def psuedoempty_columns_mask(df, col_thresh=0.1, return_dfs=None):
    """
    Create a mask for columns with a threshold of missing data with the option to treat and transform the
    original DataFrame.
    Fills the casual elements with value `3` and any affected values with `2`.

    :param df: The `DataFrame` being processed.
    :param col_thresh: The nullity score in order to drop the row. Should be a value between [0, 1].
    Higher is more likely to drop columns.
    :param return_dfs: whether to return the transformed df_ and dropped rows.
    :return: color_df or df_, removed and color_df.
    """
    color_df = pd.DataFrame(np.zeros_like(df), index=df.index, columns=df.columns).astype(int)
    color_col_mask = df.notna().mean(axis=0) <= col_thresh
    color_col_names = color_col_mask[color_col_mask].index.tolist()

    for col in color_col_names:
        color_df.loc[:, col] = df.loc[:, col].isna() + 2
    
    if return_dfs is None or not return_dfs:
        return color_df
    else:
        df_ = df.copy()
        df_ = df.loc[:, ~color_col_mask]
        removed = df.loc[:, color_col_mask]
        return df_, removed, color_df


def psuedoempty_rows_mask(df, row_thresh=0, return_dfs=None):
    """
    Create a mask for rows with a threshold of missing data with the option to treat and transform the
    original DataFrame.
    Fills the casual elements with value `3` and any affected values with `2`.

    :param df: The `DataFrame` being processed.
    :param row_thresh: The nullity score in order to drop the row. Should be a value between [0, 1].
    Higher is more likely to drop rows.
    :param return_dfs: whether to return the transformed df_ and dropped rows.
    :return: color_df or df_, removed and color_df.
    """
    color_df = pd.DataFrame(np.zeros_like(df), index=df.index, columns=df.columns).astype(int)
    color_idx_mask = df.notna().mean(axis=1) >= row_thresh

    color_df.loc[color_idx_mask, :] = df.loc[color_idx_mask, :].isna() + 2

    if return_dfs is None or not return_dfs:
        return color_df
    else:
        df_ = df.copy()
        df_ = df.loc[~color_idx_mask, :]
        removed = df.loc[color_idx_mask, :]
        return df_, removed, color_df


def interpolation_mask(df):
    """
    Create a mask for missing values, primarily to indicate interpolation or treatment.
    Fills the elements with the value `1`.

    :param df: The `DataFrame` being processed.
    :return: color_df.
    """
    color_df = df.where(df.isna(), 0).fillna(1)
    return color_df


def update_color_df(color_df, color_df_update):
    """
    Update an existing color-mask and update it with new a new color-mask.

    :param color_df: The original DataFrame.
    :param color_df_update: The DataFrame was are using to update the original DataFrame with.
    :return: color_df.
    """
    if color_df is None:
        return color_df_update

    color_df_ = color_df.copy()
    update_idx = color_df.index.isin(color_df_update.index)
    update_col = color_df.columns.isin(color_df_update.columns)
    color_df_update_ = np.where(
        color_df.loc[update_idx, update_col] < 2,
        color_df_update,
        color_df.loc[update_idx, update_col],
    )

    color_df_.loc[update_idx, update_col] = color_df_update_
    return color_df_
