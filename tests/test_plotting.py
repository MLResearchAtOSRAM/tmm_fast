import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tmm_fast import plot_stacks

RENDER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_output")


@pytest.fixture
def plot_axes():
    figure, ax = plt.subplots()
    yield figure, ax

    plt.close(figure)


@pytest.fixture
def rendered_plot_axes(plot_axes, request):
    yield plot_axes

    figure, _ = plot_axes
    os.makedirs(RENDER_DIR, exist_ok=True)
    name = "plot_" + request.node.name.replace("[", "_").replace("]", "") + ".png"
    figure.savefig(os.path.join(RENDER_DIR, name), dpi=110)


@pytest.fixture
def material_stack():
    return np.array([1.4585, 2.3403]), np.array([1e-6, 2e-6])


@pytest.fixture
def multiple_stacks():
    indexes = [np.array([1.4, 2.2]), np.array([1.6, 2.4])]
    thickness = [np.array([1e-6, 2e-6]), np.array([2e-6, 1e-6])]
    return indexes, thickness


@pytest.mark.parametrize(
    "names, expected",
    [
        pytest.param(["SiO2", "Nb2O5"], {"SiO2", "Nb2O5"}, id="material-names"),
        pytest.param(None, {"n=1.4585", "n=2.3403"}, id="refractive-index"),
    ],
)
def test_material_labels(rendered_plot_axes, material_stack, names, expected):
    _, ax = rendered_plot_axes
    indexes, thickness = material_stack
    plot_stacks(ax, indexes, thickness, names=names)

    assert {text.get_text() for text in ax.texts} == expected


def test_indexes_are_not_modified_for_multiple_stacks(
    rendered_plot_axes, multiple_stacks
):
    _, ax = rendered_plot_axes
    indexes, thickness = multiple_stacks
    before = [index.copy() for index in indexes]

    plot_stacks(ax, indexes, thickness, labels=["first", "second"], show_material=False)

    for original, current in zip(before, indexes):
        np.testing.assert_array_equal(original, current)
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["first", "second"]


@pytest.mark.parametrize(
    "indexes, thickness, kwargs, message",
    [
        pytest.param(
            [1.4, 2.2], np.array([1e-6]), {}, "one refractive index", id="layer-count"
        ),
        pytest.param(
            [1.4],
            np.array([1e-6]),
            {"names": ["air", "film"]},
            "one material name",
            id="material-name-count",
        ),
        pytest.param(
            [np.array([1.4])],
            [np.array([1e-6]), np.array([2e-6])],
            {},
            "same number of stacks",
            id="stack-count",
        ),
    ],
)
def test_plot_input_lengths_are_validated(
    plot_axes, indexes, thickness, kwargs, message
):
    _, ax = plot_axes
    with pytest.raises(ValueError, match=message):
        plot_stacks(ax, indexes, thickness, **kwargs)
