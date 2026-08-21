import os
import warnings
from contextlib import contextmanager

import matplotlib
matplotlib.use('Agg')  # never open windows, the figures are saved instead

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tmm_fast.gym_multilayerthinfilm import MultiLayerThinFilm, gym_class

# next to this file rather than in the temp directory, so the figures are easy to find when the
# tests are run from an IDE test panel; the directory is gitignored
RENDER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'render_output')
LAYOUTS = ('single', 'double')
SAVED = []

# render() calls plt.show(), which the Agg backend cannot honour
warnings.filterwarnings('ignore', message='FigureCanvasAgg is non-interactive')


@contextmanager
def figure_layout(layout):
    previous = gym_class.FIGURE_LAYOUT
    gym_class.FIGURE_LAYOUT = layout
    try:
        yield gym_class.FIGURE_GEOMETRY.get(layout)
    finally:
        gym_class.FIGURE_LAYOUT = previous


@pytest.fixture(params=LAYOUTS)
def layout(request):
    with figure_layout(request.param):
        yield request.param


def save(figure, name, layout):
    os.makedirs(RENDER_DIR, exist_ok=True)
    path = os.path.join(RENDER_DIR, '%s_%s.png' % (name, layout))
    figure.savefig(path, dpi=110)
    SAVED.append(path)
    return path


def make_env(maximum_layers=4, num_wl=32, num_angles=5):
    wl = np.linspace(400, 700, num_wl) * 1e-9
    angle = np.linspace(0, 45, num_angles)
    N = np.vstack([1.46 * np.ones(num_wl), 2.34 * np.ones(num_wl), np.ones(num_wl)])
    target = {'direction': angle,
              'spectrum': wl,
              'target': 0.5 * np.ones((num_angles, num_wl)),
              'mode': 'reflectivity'}
    return MultiLayerThinFilm(N, maximum_layers, target)


def test_step_keeps_thicknesses_scalar():
    env = make_env()
    env.reset()
    state, reward, done, _ = env.step(env.create_action(1, thickness=50e-9, is_normalized=False))
    simulation, n, d, one_hot_status = state
    assert simulation.shape == env.target.shape
    assert all(np.ndim(t) == 0 for t in d), 'thicknesses must stay scalar, got ' + str(d)
    assert one_hot_status.shape == env.observation_space.shape
    assert np.isfinite(reward)
    assert not done


def test_thickness_survives_the_round_trip():
    env = make_env()
    env.reset()
    env.step(env.create_action(2, thickness=77e-9, is_normalized=False))
    assert np.isclose(env.d[-1], 77e-9), env.d


def test_sampled_actions():
    env = make_env(maximum_layers=6)
    env.reset()
    for _ in range(6):
        _, _, done, _ = env.step(env.action_space.sample())
        if done:
            break
    assert all(np.ndim(t) == 0 for t in env.d)


def test_termination():
    env = make_env(maximum_layers=2)
    env.reset()
    for expected in (False, False, True):
        _, _, done, _ = env.step(env.create_action(1, 0.5))
        assert done == expected, 'unexpected done flag after %d layers' % len(env.layers)

    env.reset()
    _, _, done, _ = env.step(env.create_action(0, 0.5))
    assert done, 'material 0 must end the episode'


@pytest.mark.parametrize('requested', [1, 2, 3])
def test_initial_layers(requested):
    # reset() used to call random.randint(1, 0) for a single initial layer
    env = make_env(maximum_layers=4)
    env.set_initial_layers(requested)
    env.reset()
    assert 1 <= len(env.layers) <= requested, (requested, env.layers)
    assert len(env.n) == len(env.d) == len(env.layers)


def test_too_many_initial_layers():
    env = make_env(maximum_layers=2)
    try:
        env.set_initial_layers(3)
    except ValueError:
        pass
    else:
        raise AssertionError('more initial layers than maximum_layers must raise')


def test_create_stack_without_thicknesses():
    env = make_env()
    n, d, cladding = env.create_stack([1, 2], [50e-9, 80e-9])
    assert n.shape == (2, env.wl.shape[0]) and d.shape == (2,)

    # create_stack() used to call np.empty() without a shape
    n, d, cladding = env.create_stack([3])
    assert d.shape == (1,) and np.isinf(d).all(), d

    # the semi-infinite default is exactly what a cladding needs, so it has to simulate
    env.set_cladding(ambient=cladding)
    env.reset()
    state, reward, _, _ = env.step(env.create_action(1, thickness=60e-9, is_normalized=False))
    assert np.isfinite(state[0]).all() and np.isfinite(reward)


def test_unknown_figure_layout_raises():
    with figure_layout('nonsense'):
        env = make_env()
        env.reset()
        try:
            env.render()
        except ValueError as error:
            assert 'FIGURE_LAYOUT' in str(error), error
        else:
            raise AssertionError('an unknown FIGURE_LAYOUT must raise')


@pytest.mark.slow
def test_render_an_episode(layout):
    # walk a whole episode and render every intermediate stack, in both column styles, so the
    # figures show how the optical response develops as layers are added
    env = make_env(maximum_layers=4)
    env.sparse_reward = False
    env.reset()
    save(env.render_target()[0], 'target', layout)

    for material, thickness in [(2, 40e-9), (1, 95e-9), (2, 120e-9)]:
        _, reward, _, _ = env.step(env.create_action(material, thickness, is_normalized=False))
        figure, axes = env.render()
        assert len(axes) == 2, 'expected a response and a stack panel'
        assert np.isfinite(reward)
        save(figure, 'episode_%d_layers' % len(env.layers), layout)

    assert len(env.layers) == 3
    assert np.isclose(sum(env.d), 40e-9 + 95e-9 + 120e-9)
    # the layout option has to reach the figure, not just the module constant
    assert np.allclose(figure.get_size_inches(), gym_class.FIGURE_GEOMETRY[layout]['figsize']), figure.get_size_inches()
    plt.close('all')


@pytest.mark.slow
def test_render_random_episode_until_done(layout):
    env = make_env(maximum_layers=5)
    env.reset()
    done, steps = False, 0
    while not done and steps < 10:
        _, _, done, _ = env.step(env.action_space.sample())
        steps += 1
    assert done, 'a random episode must terminate within maximum_layers + 1 steps'
    save(env.render()[0], 'random_episode', layout)
    plt.close('all')


@pytest.mark.slow
@pytest.mark.parametrize('num_wl, num_angles, name', [(1, 32, 'single_wavelength'),
                                                     (32, 1, 'single_angle')])
def test_render_one_dimensional_branches(layout, num_wl, num_angles, name):
    # a single angle or a single wavelength renders a line plot instead of a heatmap; these
    # branches were unreachable while a dropped wavelength axis was rejected
    env = make_env(num_wl=num_wl, num_angles=num_angles)
    env.reset()
    env.step(env.create_action(1, thickness=60e-9, is_normalized=False))
    save(env.render()[0], 'render_' + name, layout)
    save(env.render_target()[0], 'target_' + name, layout)
    plt.close('all')


if __name__ == '__main__':
    # the render tests take their layout from a fixture, so hand the module to pytest
    code = pytest.main([__file__, '-q'])
    if os.path.isdir(RENDER_DIR):
        print('%d figures written to %s' % (len(os.listdir(RENDER_DIR)), RENDER_DIR))
    raise SystemExit(code)
