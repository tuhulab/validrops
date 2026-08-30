import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import validrops


@pytest.fixture(scope="module")
def plotted(raw_adata):
    adata = raw_adata.copy()
    validrops.validrops(adata, stage_three=False, random_state=0)
    return adata


@pytest.mark.parametrize("name", ["barcode_rank", "mito_threshold", "umi_vs_features", "coding_fraction"])
def test_plot_returns_axes(plotted, name):
    ax = getattr(validrops.pl, name)(plotted)
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel()
    assert ax.get_ylabel()
    plt.close(ax.figure)


def test_plot_accepts_an_existing_axes(plotted):
    fig, ax = plt.subplots()
    returned = validrops.pl.barcode_rank(plotted, ax=ax)
    assert returned is ax
    plt.close(fig)


def test_dead_score_requires_label_dead(plotted):
    with pytest.raises(KeyError, match="dead_score"):
        validrops.pl.dead_score(plotted)


def test_mito_threshold_draws_the_cutoff_line(plotted):
    ax = validrops.pl.mito_threshold(plotted)
    assert len(ax.lines) >= 1
    plt.close(ax.figure)
