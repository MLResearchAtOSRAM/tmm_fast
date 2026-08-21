import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np


def plot_stacks(ax, indexes, thickness, labels=None, show_material=True, names=None):
    """Plot one or more multilayer stacks, coloured by refractive index.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes that receives the plot.
    indexes : array-like or list of array-like
        Real or complex refractive indices for one stack, or one array per stack.
    thickness : array-like or list of array-like
        Layer thicknesses in metres for one stack, or one array per stack.
    labels : str or sequence of str, optional
        Labels shown below multiple stacks.
    show_material : bool, default=True
        Display the material name, or the real refractive index when no name is supplied.
    names : sequence of str or sequence of sequences, optional
        Material names in layer order. Supply one sequence per stack when plotting multiple stacks.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The modified axes.
    cmap : matplotlib.cm.ScalarMappable
        The refractive-index colour map.
    """
    multiple = isinstance(thickness, list)
    index_stacks = _as_stacks(indexes, multiple, 'indexes')
    thickness_stacks = _as_stacks(thickness, multiple, 'thickness')

    if len(index_stacks) != len(thickness_stacks):
        raise ValueError('indexes and thickness must contain the same number of stacks')
    for index, thick in zip(index_stacks, thickness_stacks):
        if index.size != thick.size:
            raise ValueError('each stack must have one refractive index per layer thickness')

    name_stacks = _names_for_stacks(names, multiple, len(index_stacks))
    for material_names, index in zip(name_stacks, index_stacks):
        if material_names is not None and len(material_names) != index.size:
            raise ValueError('names must contain one material name per layer')

    real_indexes = [np.asarray(index).real for index in index_stacks]
    all_indexes = np.concatenate(real_indexes)
    norm = colors.Normalize(vmin=np.min(all_indexes) - 1, vmax=np.max(all_indexes) + 1)
    cmap = cm.ScalarMappable(norm=norm, cmap=cm.rainbow)

    if multiple:
        stack_labels = _stack_labels(labels, len(thickness_stacks))
        max_height = max(np.sum(thick) for thick in thickness_stacks) * 1e6
        for stack, (index, thick, material_names) in enumerate(
            zip(real_indexes, thickness_stacks, name_stacks)
        ):
            _plot_stack(ax, stack * 0.4, 0.36, index, thick, material_names,
                        show_material, max_height, cmap)
        positions = np.arange(len(thickness_stacks)) * 0.4
        ax.set_xticks(positions)
        ax.set_xticklabels(stack_labels)
    else:
        total_height = np.sum(thickness_stacks[0]) * 1e6
        _plot_stack(ax, 0, 0.2, real_indexes[0], thickness_stacks[0], name_stacks[0],
                    show_material, total_height, cmap)
        ax.set_ylim(0, total_height * 1.05)
        ax.xaxis.set_visible(False)

    ax.set_ylabel(r'Thickness in $\mu$m')
    return ax, cmap


def _as_stacks(values, multiple, name):
    stacks = values if multiple else [values]
    result = [np.asarray(stack).copy() for stack in stacks]
    if any(stack.ndim != 1 for stack in result):
        raise ValueError(name + ' must contain one-dimensional layer arrays')
    return result


def _names_for_stacks(names, multiple, count):
    if names is None:
        return [None] * count
    stacks = list(names) if multiple else [list(names)]
    if len(stacks) != count:
        raise ValueError('names must contain one sequence per stack')
    return [list(stack) for stack in stacks]


def _stack_labels(labels, count):
    if labels is None:
        return [str(index + 1) for index in range(count)]
    if isinstance(labels, str):
        labels = [labels]
    if len(labels) != count:
        raise ValueError('labels must contain one label per stack')
    return labels


def _plot_stack(ax, position, width, indexes, thickness, names, show_material,
                stack_height, cmap):
    cumulative = np.cumsum(np.asarray(thickness) * 1e6)[::-1]
    layer_thicknesses = (np.asarray(thickness) * 1e6)[::-1]
    plot_indexes = indexes[::-1]
    plot_names = None if names is None else names[::-1]

    for layer, (height, layer_thickness, index) in enumerate(
        zip(cumulative, layer_thicknesses, plot_indexes)
    ):
        ax.bar(position, height, width, color=cmap.to_rgba(index))
        if show_material and layer_thickness > stack_height / 22:
            text = 'n=' + str(index) if plot_names is None else str(plot_names[layer])
            ax.text(position - width * 0.49, height - 0.008 * stack_height,
                    text, va='top', c='gray')
