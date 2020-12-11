import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import ternary
import pandas as pd


def plot_ternary(tern, scale=100, dpi=150):
    """Plot ternary ancestry fractions.

    Usses the python-ternary package.
    """
    assert tern.shape[1] == 3
    matplotlib.rcParams['figure.dpi'] = dpi
    figure, tax = ternary.figure(scale=scale)
    figure.set_size_inches(4.5, 4)

    tern_df = pd.DataFrame(tern)
    tern_df.columns = ['AA', 'AB', 'BB']
    tern_df = tern_df.div(1 / scale)

    size = 5
    alpha = 1
    tax.scatter(tern_df.values.tolist(),
                color='blue', label="", s=size, alpha=alpha, linewidth=.5)

    # Set ticks and gridlines
    tax.gridlines(color="black", multiple=scale / 4, linewidth=0.5)
    tax.ticks(axis='lbr', linewidth=1, multiple=scale / 4, offset=0.02)
    tax.boundary(linewidth=.5)
    # Remove default Matplotlib Axes
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')

    # Set Axis labels and Title
    fontsize = 10
    corner_offset = 0.35
    top_offset = 0.20
    tax.set_title("Ternary ancestry percentages\n\n", fontsize=fontsize)
    tax.right_corner_label(tern_df.columns[0], fontsize=fontsize, offset=corner_offset)
    tax.top_corner_label(tern_df.columns[1], fontsize=fontsize, offset=top_offset)
    tax.left_corner_label(tern_df.columns[2], fontsize=fontsize, offset=corner_offset)

    label_offset = 0.14
    tax.left_axis_label("BB", fontsize=fontsize, offset=label_offset)
    tax.right_axis_label("AB", fontsize=fontsize, offset=label_offset)
    tax.bottom_axis_label("AA", fontsize=fontsize, offset=label_offset)

    tax._redraw_labels()  # fix possible label drawing problems
    return(figure, tax)


def plot_Q(tern, sort=True):
    """Produce a stacked-barplot (i.e. a 'STRUCTURE' plot) from ternary ancestry fractions."""
    assert tern.shape[1] == 3
    tern_df = pd.DataFrame(tern)
    tern_df.columns = ['AA', 'AB', 'BB']
    tern_df['A'] = tern_df['AA'] + tern_df['AB'] / 2
    tern_df['B'] = tern_df['BB'] + tern_df['AB'] / 2
    Q = tern_df[['A', 'B']]
    Q = Q.div(Q.sum(1), axis=0)

    if sort:
        Q = Q.sort_values('A').reset_index(drop=True)

    fracA, fracB = Q.mean()

    figure, ax = plt.subplots()
    ax.bar(Q.index, Q['A'], width=1, label=f'A: {fracA:.1%}')
    ax.bar(Q.index, Q['B'], width=1, label=f'B: {fracB:.1%}', bottom=Q['A'])
    ax.legend()
    ax.set_xlim(0, len(Q))
    ax.set_ylim(0, 1)
    sns.despine()
    return(figure, ax)
