"""Extension of missingno to assist in how to treat, drop nad fill the missing data."""

import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib import gridspec
from scipy.cluster import hierarchy

from .utils import (
    nullity_filter, nullity_sort, key_columns_mask, psuedoempty_columns_mask,
    psuedoempty_rows_mask, interpolation_mask, update_color_df,
)


def matrix(df,
           key_cols=None, col_thresh=0.0, row_thresh=0.0,
           filter=None,
        #    n=0, p=0,
           sort=None,
           figsize=(25, 10), width_ratios=(15, 1), colors=None,
           fontsize=16, labels=None, sparkline=True, inline=False,
           freq=None, ax=None, return_dfs=None):
    """
    A matrix visualization of the nullity of the given DataFrame with added features. These features are
    checked in this order:
        1) Removes all rows where key columns contain missing data and labels the row as such. Typically,
            used for the `y` dependent variable of a dataset.
        2) Removes all columns that are empty or close to empty with defined threshold and highlights
            the rows causing the removal.
        3) Removes all rows that are empty or close to empty with defined threshold and highlights
            the columns causing the removal.
        4) Highlights all remaining missing values as missing values to be interpolated.
    
    Default functionality:
    Color: Green - (0.412, 0.686, 0.137) - Key: 0 - Data: Good
    Color: Blue  - (0, 0.439, 0.753)     - Key: 1 - Data: Missing-Interpolated.
    Color: Amber - (1.0, 0.765, 0.0)     - Key: 2 - Data: Removed-non-Causal. Note: not checked if missing.
    Color: Red   - (0.882, 0.0, 0.98)    - Key: 3 - Data: Removed-Causal.

    :param df: The `DataFrame` being mapped.
    :param key_cols: The names of the key columns. str, list or default None.
    :param col_thresh: The threshold of data to keep each column. Fraction is calculated after previous
    cleaning step. A float between 0 and 1. Default 0.0.
    :param row_thresh: The threshold of data to keep each row. Fraction is calculated after previous
    cleaning step. A float between 0 and 1. Default 0.0.
    :param filter: The filter to apply to the heatmap. Should be one of "top", "bottom", or None (default).
    :param sort: The row sort order to apply. Can be "ascending", "descending", or None.
    :param figsize: The size of the figure to display.
    :param fontsize: The figure's font size. Default to 16.
    :param labels: Whether or not to display the column names. Defaults to the underlying data labels when there are
    50 columns or less, and no labels when there are more than 50 columns.
    :param sparkline: Whether or not to display the sparkline. Defaults to True.
    :param width_ratios: The ratio of the width of the matrix to the width of the sparkline. Defaults to `(15, 1)`.
    Does nothing if `sparkline=False`.
    :param colors: The colors of the filled columns with [0, 1] RGB values. Dict, list or None. Default is None.
    :return: If `inline` is False, the underlying `matplotlib.figure` object. Else, nothing.
    """
    if isinstance(colors, list):
        color_list = colors
        color_dict = dict(zip(range(4), color_list))
    elif colors is None:
        color_list = [(105, 175, 35), (0, 112, 192), (255, 195, 0), (225, 0, 25)]
        color_list = [tuple(i / 255 for i in c) for c in color_list]
        color_dict = dict(zip(range(4), color_list))
    elif isinstance(colors, dict):
        color_dict = colors
    else:
        raise Exception('`colors` must be list, dict or None.')

    df = nullity_filter(df, filter=filter)
    df = nullity_sort(df, sort=sort, axis='columns')

    color_df = None
    color_df_update = key_columns_mask(df, key_cols=key_cols, return_dfs=return_dfs)
    color_df = update_color_df(color_df, color_df_update)

    color_df_update = psuedoempty_columns_mask(
        df, col_thresh=col_thresh, return_dfs=return_dfs
    )
    color_df = update_color_df(color_df, color_df_update)

    color_df_update = psuedoempty_rows_mask(
        df, row_thresh=row_thresh, return_dfs=return_dfs
    )
    color_df = update_color_df(color_df, color_df_update)

    color_df_update = interpolation_mask(df)
    color_df = update_color_df(color_df, color_df_update)

    height = df.shape[0]
    width = df.shape[1]

    # z is the color-mask array, g is a NxNx3 matrix. Apply the z color-mask to set the RGB of each pixel.
    z = color_df
    g = np.zeros((height, width, 3), dtype=np.float32)

    for i in range(4):
        g[(z == i).values] = color_dict[i]

    # Set up the matplotlib grid layout. A unary subplot if no sparkline, a left-right splot if yes sparkline.
    if ax is None:
        plt.figure(figsize=figsize)
        if sparkline:
            gs = gridspec.GridSpec(1, 2, width_ratios=width_ratios)
            gs.update(wspace=0.08)
            ax1 = plt.subplot(gs[1])
        else:
            gs = gridspec.GridSpec(1, 1)
        ax0 = plt.subplot(gs[0])
    else:
        if sparkline is not False:
            warnings.warn(
                "Plotting a sparkline on an existing axis is not currently supported. "
                "To remove this warning, set sparkline=False."
            )
            sparkline = False
        ax0 = ax

    # Create the nullity plot.
    ax0.imshow(g, interpolation='none')

    # Remove extraneous default visual elements.
    ax0.set_aspect('auto')
    ax0.grid(b=False)
    ax0.xaxis.tick_top()
    ax0.xaxis.set_ticks_position('none')
    ax0.yaxis.set_ticks_position('none')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)

    # Set up and rotate the column ticks. The labels argument is set to None by default. If the user specifies it in
    # the argument, respect that specification. Otherwise display for <= 50 columns and do not display for > 50.
    if labels or (labels is None and len(df.columns) <= 50):
        ha = 'left'
        ax0.set_xticks(list(range(0, width)))
        ax0.set_xticklabels(list(df.columns), rotation=45, ha=ha, fontsize=fontsize)
    else:
        ax0.set_xticks([])

    # Adds Timestamps ticks if freq is not None, else set up the two top-bottom row ticks.
    if freq:
        ts_list = []

        if type(df.index) == pd.PeriodIndex:
            ts_array = pd.date_range(df.index.to_timestamp().date[0],
                                     df.index.to_timestamp().date[-1],
                                     freq=freq).values

            ts_ticks = pd.date_range(df.index.to_timestamp().date[0],
                                     df.index.to_timestamp().date[-1],
                                     freq=freq).map(lambda t:
                                                    t.strftime('%Y-%m-%d'))

        elif type(df.index) == pd.DatetimeIndex:
            ts_array = pd.date_range(df.index[0], df.index[-1],
                                     freq=freq).values

            ts_ticks = pd.date_range(df.index[0], df.index[-1],
                                     freq=freq).map(lambda t:
                                                    t.strftime('%Y-%m-%d'))
        else:
            raise KeyError('Dataframe index must be PeriodIndex or DatetimeIndex.')
        try:
            for value in ts_array:
                ts_list.append(df.index.get_loc(value))
        except KeyError:
            raise KeyError('Could not divide time index into desired frequency.')

        ax0.set_yticks(ts_list)
        ax0.set_yticklabels(ts_ticks, fontsize=int(fontsize / 16 * 20), rotation=0)
    else:
        ax0.set_yticks([0, df.shape[0] - 1])
        ax0.set_yticklabels([1, df.shape[0]], fontsize=int(fontsize / 16 * 20), rotation=0)

    # Create the inter-column vertical grid.
    in_between_point = [x + 0.5 for x in range(0, width - 1)]
    for in_between_point in in_between_point:
        ax0.axvline(in_between_point, linestyle='-', color='white')

    if sparkline:
        # Calculate row-wise completeness for the sparkline.
        completeness_srs = df.notnull().astype(bool).sum(axis=1)
        x_domain = list(range(0, height))
        y_range = list(reversed(completeness_srs.values))
        min_completeness = min(y_range)
        max_completeness = max(y_range)
        min_completeness_index = y_range.index(min_completeness)
        max_completeness_index = y_range.index(max_completeness)

        # Set up the sparkline, remove the border element.
        ax1.grid(b=False)
        ax1.set_aspect('auto')
        # GH 25
        if int(mpl.__version__[0]) <= 1:
            ax1.set_axis_bgcolor((1, 1, 1))
        else:
            ax1.set_facecolor((1, 1, 1))
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.set_ymargin(0)

        # Plot sparkline---plot is sideways so the x and y axis are reversed.
        ax1.plot(y_range, x_domain, color=color_dict[1])

        if labels:
            # Figure out what case to display the label in: mixed, upper, lower.
            label = 'Data Completeness'
            if str(df.columns[0]).islower():
                label = label.lower()
            if str(df.columns[0]).isupper():
                label = label.upper()

            # Set up and rotate the sparkline label.
            ha = 'left'
            ax1.set_xticks([min_completeness + (max_completeness - min_completeness) / 2])
            ax1.set_xticklabels([label], rotation=45, ha=ha, fontsize=fontsize)
            ax1.xaxis.tick_top()
            ax1.set_yticks([])
        else:
            ax1.set_xticks([])
            ax1.set_yticks([])

        # Add maximum and minimum labels, circles.
        ax1.annotate(max_completeness,
                     xy=(max_completeness, max_completeness_index),
                     xytext=(max_completeness + 2, max_completeness_index),
                     fontsize=int(fontsize / 16 * 14),
                     va='center',
                     ha='left')
        ax1.annotate(min_completeness,
                     xy=(min_completeness, min_completeness_index),
                     xytext=(min_completeness - 2, min_completeness_index),
                     fontsize=int(fontsize / 16 * 14),
                     va='center',
                     ha='right')

        ax1.set_xlim([min_completeness - 2, max_completeness + 2])  # Otherwise the circles are cut off.
        ax1.plot([min_completeness], [min_completeness_index], '.', color=color_dict[1], markersize=10.0)
        ax1.plot([max_completeness], [max_completeness_index], '.', color=color_dict[1], markersize=10.0)

        # Remove tick mark (only works after plotting).
        ax1.xaxis.set_ticks_position('none')

    if inline:
        warnings.warn(
            "The 'inline' argument has been deprecated, and will be removed in a future version "
            "of missingno."
        )
        plt.show()
    else:
        return ax0


# def bar(df, figsize=(24, 10), fontsize=16, labels=None, log=False, color='dimgray', inline=False,
#         filter=None, n=0, p=0, sort=None, ax=None):
#     """
#     A bar chart visualization of the nullity of the given DataFrame.

#     :param df: The input DataFrame.
#     :param log: Whether or not to display a logorithmic plot. Defaults to False (linear).
#     :param filter: The filter to apply to the heatmap. Should be one of "top", "bottom", or None (default).
#     :param n: The cap on the number of columns to include in the filtered DataFrame.
#     :param p: The cap on the percentage fill of the columns in the filtered DataFrame.
#     :param sort: The column sort order to apply. Can be "ascending", "descending", or None.
#     :param figsize: The size of the figure to display.
#     :param fontsize: The figure's font size. This default to 16.
#     :param labels: Whether or not to display the column names. Would need to be turned off on particularly large
#     displays. Defaults to True.
#     :param color: The color of the filled columns. Default to the RGB multiple `(0.25, 0.25, 0.25)`.
#     :return: If `inline` is False, the underlying `matplotlib.figure` object. Else, nothing.
#     """
#     df = nullity_filter(df, filter=filter, n=n, p=p)
#     df = nullity_sort(df, sort=sort, axis='rows')
#     nullity_counts = len(df) - df.isnull().sum()

#     if ax is None:
#         ax1 = plt.gca()
#     else:
#         ax1 = ax
#         figsize = None  # for behavioral consistency with other plot types, re-use the given size

#     (nullity_counts / len(df)).plot.bar(
#         figsize=figsize, fontsize=fontsize, log=log, color=color, ax=ax1
#     )

#     axes = [ax1]

#     # Start appending elements, starting with a modified bottom x axis.
#     if labels or (labels is None and len(df.columns) <= 50):
#         ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize)

#         # Create the numerical ticks.
#         ax2 = ax1.twinx()
#         axes.append(ax2)
#         if not log:
#             ax1.set_ylim([0, 1])
#             ax2.set_yticks(ax1.get_yticks())
#             ax2.set_yticklabels([int(n*len(df)) for n in ax1.get_yticks()], fontsize=fontsize)
#         else:
#             # For some reason when a logarithmic plot is specified `ax1` always contains two more ticks than actually
#             # appears in the plot. The fix is to ignore the first and last entries. Also note that when a log scale
#             # is used, we have to make it match the `ax1` layout ourselves.
#             ax2.set_yscale('log')
#             ax2.set_ylim(ax1.get_ylim())
#             ax2.set_yticklabels([int(n*len(df)) for n in ax1.get_yticks()], fontsize=fontsize)
#     else:
#         ax1.set_xticks([])

#     # Create the third axis, which displays columnar totals above the rest of the plot.
#     ax3 = ax1.twiny()
#     axes.append(ax3)
#     ax3.set_xticks(ax1.get_xticks())
#     ax3.set_xlim(ax1.get_xlim())
#     ax3.set_xticklabels(nullity_counts.values, fontsize=fontsize, rotation=45, ha='left')
#     ax3.grid(False)

#     for ax in axes:
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.spines['bottom'].set_visible(False)
#         ax.spines['left'].set_visible(False)
#         ax.xaxis.set_ticks_position('none')
#         ax.yaxis.set_ticks_position('none')

#     if inline:
#         warnings.warn(
#             "The 'inline' argument has been deprecated, and will be removed in a future version "
#             "of missingno."
#         )
#         plt.show()
#     else:
#         return ax1


# def dendrogram(df, method='average',
#                filter=None, n=0, p=0,
#                orientation=None, figsize=None,
#                fontsize=16, inline=False, ax=None
#                ):
#     """
#     Fits a `scipy` hierarchical clustering algorithm to the given DataFrame's variables and visualizes the results as
#     a `scipy` dendrogram.

#     The default vertical display will fit up to 50 columns. If more than 50 columns are specified and orientation is
#     left unspecified the dendrogram will automatically swap to a horizontal display to fit the additional variables.

#     :param df: The DataFrame whose completeness is being dendrogrammed.
#     :param method: The distance measure being used for clustering. This is a parameter that is passed to 
#     `scipy.hierarchy`.
#     :param filter: The filter to apply to the heatmap. Should be one of "top", "bottom", or None (default).
#     :param n: The cap on the number of columns to include in the filtered DataFrame.
#     :param p: The cap on the percentage fill of the columns in the filtered DataFrame.
#     :param figsize: The size of the figure to display. This is a `matplotlib` parameter which defaults to `(25, 10)`.
#     :param fontsize: The figure's font size.
#     :param orientation: The way the dendrogram is oriented. Defaults to top-down if there are less than or equal to 50
#     columns and left-right if there are more.
#     :param inline: Whether or not the figure is inline. If it's not then instead of getting plotted, this method will
#     return its figure.
#     :return: If `inline` is False, the underlying `matplotlib.figure` object. Else, nothing.
#     """
#     if not figsize:
#         if len(df.columns) <= 50 or orientation == 'top' or orientation == 'bottom':
#             figsize = (25, 10)
#         else:
#             figsize = (25, (25 + len(df.columns) - 50) * 0.5)

#     if ax is None:
#         plt.figure(figsize=figsize)
#         ax0 = plt.gca()
#     else:
#         ax0 = ax

#     df = nullity_filter(df, filter=filter, n=n, p=p)

#     # Link the hierarchical output matrix, figure out orientation, construct base dendrogram.
#     x = np.transpose(df.isnull().astype(int).values)
#     z = hierarchy.linkage(x, method)

#     if not orientation:
#         if len(df.columns) > 50:
#             orientation = 'left'
#         else:
#             orientation = 'bottom'

#     hierarchy.dendrogram(
#         z,
#         orientation=orientation,
#         labels=df.columns.tolist(),
#         distance_sort='descending',
#         link_color_func=lambda c: 'black',
#         leaf_font_size=fontsize,
#         ax=ax0
#     )

#     # Remove extraneous default visual elements.
#     ax0.set_aspect('auto')
#     ax0.grid(b=False)
#     if orientation == 'bottom':
#         ax0.xaxis.tick_top()
#     ax0.xaxis.set_ticks_position('none')
#     ax0.yaxis.set_ticks_position('none')
#     ax0.spines['top'].set_visible(False)
#     ax0.spines['right'].set_visible(False)
#     ax0.spines['bottom'].set_visible(False)
#     ax0.spines['left'].set_visible(False)
#     ax0.patch.set_visible(False)

#     # Set up the categorical axis labels and draw.
#     if orientation == 'bottom':
#         ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(), rotation=45, ha='left')
#     elif orientation == 'top':
#         ax0.set_xticklabels(ax0.xaxis.get_majorticklabels(), rotation=45, ha='right')
#     if orientation == 'bottom' or orientation == 'top':
#         ax0.tick_params(axis='y', labelsize=int(fontsize / 16 * 20))
#     else:
#         ax0.tick_params(axis='x', labelsize=int(fontsize / 16 * 20))

#     if inline:
#         warnings.warn(
#             "The 'inline' argument has been deprecated, and will be removed in a future version "
#             "of missingno."
#         )
#         plt.show()
#     else:
#         return ax0
