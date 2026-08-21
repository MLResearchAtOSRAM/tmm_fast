import random
import numpy as np
import gymnasium
from gymnasium import spaces
from ..vectorized_tmm_dispersive_multistack import coh_vec_tmm_disp_mstack as tmm
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D

# Colormap and number of axis ticks used by the heatmaps in render() and render_target()
HEATMAP_CMAP = 'viridis'
HEATMAP_TICKS = 10

# Figure geometry of render() and render_target(). 'single' fits one column of a two column
# paper and stacks the panels, 'double' spans both columns and puts them side by side.
FIGURE_LAYOUT = 'single'
FIGURE_GEOMETRY = {
    'single': {'figsize': (3.6, 5.6), 'nrows': 2, 'ncols': 1,
               'legend_columns': 3, 'stack_yaxis_right': False},
    'double': {'figsize': (7.2, 3.4), 'nrows': 1, 'ncols': 2,
               'legend_columns': 6, 'stack_yaxis_right': True},
}


def _geometry():
    if FIGURE_LAYOUT not in FIGURE_GEOMETRY:
        raise ValueError("FIGURE_LAYOUT must be one of '" + "', '".join(sorted(FIGURE_GEOMETRY)) + "'")
    return FIGURE_GEOMETRY[FIGURE_LAYOUT]


def _new_figure(ratios=None):
    """
    Creates the two panel figure that render() and render_target() draw into, arranged as
    FIGURE_LAYOUT asks. ratios weights the panels along the arrangement, and the constrained
    layout is what keeps titles, labels and the legend from colliding.
    """
    geometry = _geometry()
    keywords = {'nrows': geometry['nrows'], 'ncols': geometry['ncols'],
                'figsize': geometry['figsize'], 'layout': 'constrained'}
    if ratios is not None:
        keywords['height_ratios' if geometry['nrows'] > 1 else 'width_ratios'] = ratios
    return plt.subplots(**keywords)



def _heatmap(ax, data, x_range, y_range, vmin=None, vmax=None, cmap=HEATMAP_CMAP, cbar=True):
    """
    Draws a 2d array as a heatmap on ax, with the first row at the top.

    Parameters:
    -----------
    ax : matplotlib axes object
        Axes to draw into
    data : np.array of shape [D x S]
        Values to plot, e.g. a reflectivity over angle (rows) and wavelength (columns)
    x_range, y_range : tuple of two floats
        (min, max) of the quantity the respective axis represents; both axes are labelled
        with HEATMAP_TICKS equidistant values taken from these ranges
    vmin, vmax : float or None
        Limits of the color scale, None autoscales to the data
    cbar : bool
        Whether to attach a colorbar to ax

    Returns:
    --------
    image : matplotlib AxesImage
        The drawn image, e.g. to attach a colorbar to later on
    """
    num_rows, num_columns = data.shape
    image = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto', origin='upper',
                      interpolation='nearest', extent=(0, num_columns, num_rows, 0))
    if cbar:
        ax.figure.colorbar(image, ax=ax)
    ax.set_xticks(np.linspace(0, num_columns, HEATMAP_TICKS))
    ax.set_yticks(np.linspace(0, num_rows, HEATMAP_TICKS))
    ax.set_xticklabels(np.linspace(x_range[0], x_range[1], HEATMAP_TICKS, dtype=int), rotation=45, ha='right')
    ax.set_yticklabels(np.linspace(y_range[0], y_range[1], HEATMAP_TICKS, dtype=int), rotation=0)
    return image


class MultiLayerThinFilm(gymnasium.Env):
    def __init__(self, 
                 N:np.array, 
                 maximum_layers:int, 
                 target:dict, 
                 weights:np.array=None, 
                 normalization:bool=True, 
                 sparse_reward:bool=True, 
                 substrate:dict=None, 
                 ambient:dict=None, 
                 relative_reward:bool=True, 
                 max_thickness:float=150e-9, 
                 min_thickness:float=10e-9, 
                 work_path:str='')->None:
        """
        Initialize a new environment for multi-layer thin-film (MLTF) optimization.
        Each layer is determined by its (dispersive) refractive index and a thickness.
        Thus, aside from choosing the material of a next layer to place, a reinforcement learning agent
        must also assign a thickness to this layer. This formulation allows to interpret the subsequent stacking of
        layers as a parameterized Markov decision process. (See publication for details)

        Contributors: Heribert Wankerl, Alexander Luce, Maike Stern
        (Feel free to add your name to the list in case you made major contributions or changes)
        Parameters:
        -----------
        N : np.array of shape [M x S]
            where M is the number of available materials and S is the number of supporting points of the spectrum
            N holds the (dispersive, complex) refractive indices of the available materials
        maximum_layers : integer
            maximum_layers defines the maximum number of layers to stack
        target : dictionary
            with keys 'target', 'direction', 'spectrum' and 'mode'
            target['direction'] holds the angles [deg, °] under consideration and is of shape D
            target['spectrum'] holds the spectrum [m] under consideration and is of shape S
            target['target'] holds the pixel-wise target reflectivity of a MLTF and is of shape [D x S]
            target['mode'] states whether to use 'reflectivity' or 'transmittivity'
        weights : np.array of same shape [D x S]
            This array allows to steer the pixels relative influence on the optimization/reward
        normalization : bool
            Determines whether to exponentially transform the reward or not (Look publication for details)
        sparse_reward : bool
            Determines whether to simulate and reward each intermediate stack or only the final stack
        substrate : dictionary or None
            Holds the (dispersive, complex) refractive indices of the substrate materials in the rows of
            substrate['n'] which is of shape np.array of shape [Sub x S] . Sub is the number of materials that form
            the substrate. substrate['d'] is of shape Sub and holds the corresponding thicknesses of each layer.
            If substrate is None (default) it is set to vacuum of infinite thickness.
        ambient : dictionary or None
            Holds the (dispersive, complex) refractive indices of the ambient materials in the rows of
            ambient['n'] which is of shape np.array of shape [Am x S] . Am is the number of materials that form
            the ambient. ambient['d'] is of shape Am and holds the corresponding thicknesses of each layer.
            If ambient is None (default) it is set to vacuum of infinite thickness.
        relative_reward : bool
            Impact only if sparse_reward is False. Determines whether the provided reward signal for a stack is
            computed independently (False) or as difference between two subsequent rewards (True), i.e. improvements
            achieved by an action are measured by the reward in the latter case.
        max_thickness : float
            Determines the maximum layer thickness in meter.
        min_thickness : float
            Determines the minimum layer thickness in meter.
        work_path : str
            Path to working directory e.g. to save images
        """

        self.N = N
        self.maximum_layers = maximum_layers
        # target-related:
        # wavelength range
        self.wl = target['spectrum']
        # mode: 'reflectivity' or 'transmittivity'
        if target['mode'] == 'transmittivity' or target['mode'] == 'reflectivity':
            self.mode = target['mode']
        else:
            self.mode = 'reflectivity'
            print('Invalid mode -> set to reflectivity!')
        # angel range
        self.angle = target['direction']
        # desired value for reflectivity (pixelwise)
        self.target = target['target']
        if weights is None:
            self.weights = np.ones_like(self.target)
        # reward computation:
        self.normalization = normalization
        self.sparse_reward = sparse_reward
        # cladding:
        if substrate is not None:
            self.n_substrate = substrate['n']
            self.d_substrate = substrate['d']
        else:
            self.n_substrate = np.ones((1, self.wl.shape[0]))
            self.d_substrate = np.array([np.inf])
            print('--- substrate is set to vacuum of infinite thickness ---')
        if ambient is not None:
            self.n_ambient = ambient['n']
            self.d_ambient = ambient['d']
        else:
            self.n_ambient = np.ones((1, self.wl.shape[0]))
            self.d_ambient = np.array([np.inf])
            print('--- ambient is set to vacuum of infinite thickness ---')
        if np.iscomplex(self.n_substrate[0, :]).any():
            self.n_substrate[0, :] = np.real(self.n_substrate[0, :])
            print('n_substrate must feature real-valued refractive indicies in first layer for computational/physical reasons (TMM); adopted via np.real()')
        if np.iscomplex(self.n_ambient[-1, :]).any():
            self.n_ambient[-1, :] = np.real(self.n_ambient[-1, :])
            print('n_ambient must feature real-valued refractive indicies in last layer for computational/physical reasons (TMM); adopted via np.real()')
        assert not np.iscomplex(self.n_substrate[0, :]).any(), 'n_substrate must feature real-valued refractive indicies in first layer for computational/physical reasons (TMM)..'
        assert not np.iscomplex(self.n_ambient[-1, :]).any(), 'n_ambient must feature real-valued refractive indicies in last layer for computational/physical reasons (TMM)..'
        self.d_substrate = self.d_substrate.reshape(-1, 1)
        self.d_ambient = self.d_ambient.reshape(-1, 1)

        self.work_path = work_path
        # borders for thicknesses:
        self.max_thickness = max_thickness
        self.min_thickness = min_thickness

        # initialization for some attributes:
        self.number_of_materials = N.shape[0]
        self.simulation = np.nan * np.zeros_like(self.target)
        self.reward = None
        self.old_reward = None
        self.relative_reward = relative_reward
        self._reward_track = []
        self.layers = []
        self._initial_nmb_layers = 0

        self.n = []
        self.d = []
        self.f = None
        self.axs = None
        # OpenAI/Farama-Foundation gymnasium related settings:
        # action space:
        space_list = [spaces.Discrete((self.number_of_materials + 1))]
        for space in range(self.number_of_materials + 1):
            space_list.append(spaces.Box(low=0, high=1, shape=(1,)))
        self.action_space = spaces.Tuple(space_list)

        # simulation state space:
        self.observation_space = spaces.Box(low=0, high=1, shape=((self.number_of_materials + 1)*maximum_layers, ), dtype=np.float64)
        if weights is None:
            self.weights = np.ones_like(self.target)
        else:
            self.weights = weights
            assert weights.shape[0] == self.target.shape[0] and weights.shape[1] == self.target.shape[1], 'Shape of weights and target must coincide!'
            assert np.all(weights >= 0), 'weights are supposed to be non-negative!'
            if np.all(weights == 0):
                self.weights = np.ones_like(self.target)
                print('All weights were zero -> if nothing is important quit optimization; we set each weight to one for you ;)...')
        self.weights = self.weights / np.max(self.weights)
        assert self.N.shape == tuple([self.number_of_materials, self.wl.shape[0]]), 'N does not match with target!'

    def set_cladding(self, substrate=None, ambient=None):
        """
        Sets the substrate and ambient that clad every stack. Both default to vacuum of infinite
        thickness. The outermost refractive indices are forced to be real, as the transfer matrix
        method requires.

        Parameters:
        -----------
        substrate, ambient : dict or None
            With 'n' of shape [layers x S] and 'd' of shape [layers], as built by create_stack()
        """
        if substrate is not None:
            self.n_substrate = substrate['n']
            self.d_substrate = substrate['d']
        else:
            self.n_substrate = np.ones((1, self.wl.shape[0]))
            self.d_substrate = np.array([np.inf]).squeeze()
        if ambient is not None:
            self.n_ambient = ambient['n']
            self.d_ambient = ambient['d']
        else:
            self.n_ambient = np.ones((1, self.wl.shape[0]))
            self.d_ambient = np.array([np.inf]).squeeze()
            print('--- ambient is set to vacuum of infinite thickness ---')
        if np.iscomplex(self.n_substrate[0, :]).any():
            self.n_substrate[0, :] = np.real(self.n_substrate[0, :])
            print(
                'n_substrate must feature real-valued refractive indicies in first layer for computational/physical reasons (TMM); adopted via np.real()')
        if np.iscomplex(self.n_ambient[-1, :]).any():
            self.n_ambient[-1, :] = np.real(self.n_ambient[-1, :])
            print(
                'n_ambient must feature real-valued refractive indicies in last layer for computational/physical reasons (TMM); adopted via np.real()')
        assert not np.iscomplex(self.n_substrate[0, :]).any(), 'n_substrate must feature real-valued refractive indicies in first layer for computational/physical reasons (TMM)..'
        assert not np.iscomplex(self.n_ambient[-1, :]).any(), 'n_ambient must feature real-valued refractive indicies in last layer for computational/physical reasons (TMM)..'
        self.d_substrate = self.d_substrate.reshape(-1, 1)
        self.d_ambient = self.d_ambient.reshape(-1, 1)
        print('cladding set....')

    def step(self, action):
        """
        Stacks one layer of a given material and thickness on top of the current stack.

        Parameters:
        -----------
        action : tuple
            action[0] is the material as an integer, where 0 ends the episode and i > 0 selects
            row i - 1 of N. action[1] is the normalized thickness in [0, 1], mapped onto
            [min_thickness, max_thickness].

        Returns:
        --------
        state : list
            [simulation, n, d, one_hot_status]; simulation is the optical response of shape
            [D x S], n the refractive indices of the stacked layers, d their thicknesses in
            meter, and one_hot_status the flat encoding described in one_hot_layer_status()
        reward : float
            Rates the stack against the target, see reward_func(). Zero unless the episode ended
            or sparse_reward is False, and a difference between successive rewards if
            relative_reward is set
        done : bool
            Whether stacking ended, either because material 0 was chosen or because
            maximum_layers is reached
        info : list
            Always empty. Note that this is the pre-gymnasium four value signature; gymnasium
            expects (observation, reward, terminated, truncated, info)
        """

        done = False
        self.old_reward = self.reward
        if action[0] == 0 or len(self.layers) >= self.maximum_layers:
            done = True
        else:
            self.layers.append(int(action[0]))
            n_layer = self.N[int(action[0] - 1), :].reshape(1, -1)
            # action[1] is a shape [1] array, both from create_action() and from the Box
            # action space, and self.d must hold plain floats
            thickness = np.asarray(action[1]).item()
            d_layer = (self.max_thickness - self.min_thickness) * thickness + self.min_thickness
            self.n.append(n_layer)
            self.d.append(d_layer)
        cladded_n, cladded_d = self.stack_layers()
        self.simulation = self.simulate(cladded_n, cladded_d)
        self.reward = 0
        if done or not self.sparse_reward:
            self.reward, mse = self.reward_func(self.simulation, self.target, self.weights, self.baseline_mse, self.normalization)
            if done:
                self._reward_track.append(mse)  # track reward to compute baseline
        if np.all(np.isnan(self.simulation)):
            print('All simulated values in TMM are NaN!')
        one_hot_status = self.one_hot_layer_status()
        if self.relative_reward and not self.sparse_reward and self.old_reward is not None and self.reward is not None:
            relative_reward = self.reward - self.old_reward
            handback_reward = relative_reward
        else:
            handback_reward = self.reward
        return [self.simulation, self.n, self.d, one_hot_status], handback_reward, done, []

    def one_hot_layer_status(self):
        """
        Encodes the whole stack as one flat vector of length maximum_layers * (M + 1). Each layer
        owns one partition of M + 1 entries, in which the entry at its material index holds the
        normalized thickness and the rest are zero.
        """
        one_hot_vectors = []
        for layer in range(self.maximum_layers):
            one_hot_vector = np.zeros((self.number_of_materials + 1))
            if layer < len(self.layers):
                one_hot_vector[int(self.layers[layer])] = 1* self.normalize_thickness(self.d[layer])
            one_hot_vectors.append(one_hot_vector)
        one_hot_vectors = np.hstack(one_hot_vectors)
        return one_hot_vectors

    def denormalize_thickness(self, t):
        """Maps a normalized thickness in [0, 1] onto [min_thickness, max_thickness]."""
        t = (self.max_thickness - self.min_thickness) * t + self.min_thickness
        return t

    def normalize_thickness(self, t):
        """Maps a thickness in meter onto [0, 1], the inverse of denormalize_thickness()."""
        t = (t - self.min_thickness) / (self.max_thickness - self.min_thickness)
        return t

    def reset(self):
        """
        Resets the environment to an empty stack, or to a random one if set_initial_layers() was
        used, and simulates it.

        Returns:
        --------
        state : list
            [simulation, n, d, one_hot_status], as returned by step()
        reward : float
            Zero unless sparse_reward is False, in which case the initial stack is rated
        info, extra : list
            Both always empty. Note that gymnasium expects reset(seed, options) returning
            (observation, info)
        """
        self.layers = []
        self.n = []
        self.d = []
        if self._initial_nmb_layers > 0:
            num_layers = random.randint(1, self._initial_nmb_layers)
            for _ in range(num_layers):
                rnd_material_idx = random.randint(0, self.number_of_materials-1)
                rnd_material_d = random.uniform(0, 1)
                self.layers.append(rnd_material_idx+1)
                n_layer = self.N[rnd_material_idx].reshape(1, -1)
                d_layer = (self.max_thickness - self.min_thickness) * rnd_material_d + self.min_thickness
                self.n.append(n_layer)
                self.d.append(d_layer)
        cladded_n, cladded_d = self.stack_layers()
        self.simulation = self.simulate(cladded_n, cladded_d)

        self.reward = 0
        if not self.sparse_reward:
            self.reward, _ = self.reward_func(self.simulation, self.target, self.weights, self.baseline_mse, self.normalization)
        one_hot_status = self.one_hot_layer_status()
        return [self.simulation, self.n, self.d, one_hot_status], self.reward, [], []

    def render(self, conduct_simulation=True, scale=False):
        """
        Renders the current stack next to its optical response and a material legend.

        Parameters:
        -----------
        conduct_simulation : bool
            Whether to simulate the current stack first instead of plotting the stored result
        scale : bool
            Whether to scale the color range to the simulated values rather than to [0, 1]

        Returns:
        --------
        list
            [figure, axes], so that the plot can be modified further
        """
        colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
        assert self.N.shape[0] <= len(colors), 'Not enough colors to illustrate all materials in N!'
        assert self.wl.shape[0] > 1 or self.angle.shape[0] > 1, 'No rendering for single wavelenght and single direction!'
        geometry = _geometry()

        # a fresh figure needs the colorbar and the material legend as well; on a redraw the
        # panels are cleared but both of those live outside them and survive
        fresh = self.f is None or not plt.fignum_exists(self.f.number)
        if fresh:
            self.f, self.axs = _new_figure(ratios=(2, 1))
            handles = [Line2D([0], [0], color=colors[material], lw=6,
                              label='Material ' + str(material + 1))
                       for material in range(self.N.shape[0])]
            self.f.legend(handles=handles, loc='outside lower center', frameon=False,
                          fontsize='small', ncol=min(len(handles), geometry['legend_columns']))
        else:
            for ax in self.axs:
                ax.clear()

        if conduct_simulation:
            cladded_n, cladded_d = self.stack_layers()
            self.simulation = self.simulate(cladded_n, cladded_d)
        self.reward, _ = self.reward_func(self.simulation, self.target, self.weights, self.baseline_mse, self.normalization)
        if scale:
            min_val = np.min(self.simulation)
            max_val = np.max(self.simulation)
        else:
            min_val = 0
            max_val = 1
        response = self.mode.capitalize()

        # the optical response, as a spectrum, over angle, or as a heatmap over both
        if self.angle.shape[0] == 1:
            self.axs[0].plot(self.wl * 10 ** 9, self.simulation.squeeze())
            self.axs[0].set_xlabel('Wavelength [nm]')
            self.axs[0].set_ylabel(response + ' [1]')
            self.axs[0].set_ylim(0, 1.05)
            self.axs[0].set_title(response + ' at ' + str(np.round(self.angle[0], 1)) + '°')
        elif self.wl.shape[0] == 1:
            self.axs[0].plot(self.angle, self.simulation.squeeze())
            self.axs[0].set_xlabel('Angle [deg, °]')
            self.axs[0].set_ylabel(response + ' [1]')
            self.axs[0].set_ylim(0, 1.05)
            self.axs[0].set_title(response + ' at ' + str(np.round(self.wl[0] * 10 ** 9, 1)) + ' nm')
        else:
            _heatmap(self.axs[0], self.simulation,
                     x_range=(np.min(self.wl * 10 ** 9), np.max(self.wl * 10 ** 9)),
                     y_range=(np.min(self.angle), np.max(self.angle)),
                     vmin=min_val, vmax=max_val, cbar=fresh)
            self.axs[0].set_xlabel('Wavelength [nm]')
            self.axs[0].set_ylabel('Angle [deg, °]')
            self.axs[0].set_title(response)

        # the stack itself, one bar segment per layer coloured by its material
        for layer, material in enumerate(self.layers):
            self.axs[1].bar(0, self.d[layer], 0.6, bottom=np.sum(self.d[:layer]),
                            color=colors[int(material) - 1])
        self.axs[1].set_xlim(-0.5, 0.5)
        self.axs[1].set_xticks([0])
        self.axs[1].set_xticklabels([str(self.num_layers) + ' layers'])
        self.axs[1].set_ylabel('Thickness [m]')
        self.axs[1].ticklabel_format(axis='y', style='sci', scilimits=(-9, -9))
        self.axs[1].yaxis.grid(linestyle='dotted')
        if geometry['stack_yaxis_right']:
            self.axs[1].yaxis.tick_right()
            self.axs[1].yaxis.set_label_position('right')

        # the reward belongs to the figure, which keeps it out of the panel titles
        self.f.suptitle('Reward = ' + str(np.round(self.reward, 4)))
        plt.show(block=False)
        plt.pause(0.1)
        return [self.f, self.axs]

    def render_target(self):
        """
        Renders the target response next to the weights that steer its influence on the reward.

        Returns:
        --------
        list
            [figure, axes], so that the plot can be modified further
        """
        assert self.wl.shape[0] > 1 or self.angle.shape[0] > 1, 'No rendering for single wavelenght and single direction!'
        f_target, axs_target = _new_figure()

        if self.angle.shape[0] == 1:
            suffix = ' at ' + str(np.round(self.angle[0], 1)) + '°'
            x, x_label = self.wl * 10 ** 9, 'Wavelength [nm]'
        elif self.wl.shape[0] == 1:
            suffix = ' at ' + str(np.round(self.wl[0] * 10 ** 9, 1)) + ' nm'
            x, x_label = self.angle, 'Angle [deg, °]'
        else:
            x = None

        if x is not None:
            axs_target[0].plot(x, self.target.squeeze())
            axs_target[0].set_ylabel(self.mode + ' [1]')
            axs_target[0].set_ylim(0, 1.05)
            axs_target[0].set_title('Target' + suffix)
            axs_target[1].plot(x, self.weights.squeeze())
            axs_target[1].set_ylabel('Weight [1]')
            axs_target[1].set_ylim(0, 1.05 * np.max(self.weights))
            axs_target[1].set_title('Weights' + suffix)
            for ax in axs_target:
                ax.set_xlabel(x_label)
        else:
            wl_range = (np.min(self.wl * 10 ** 9), np.max(self.wl * 10 ** 9))
            angle_range = (np.min(self.angle), np.max(self.angle))
            _heatmap(axs_target[0], self.target, x_range=wl_range, y_range=angle_range, vmin=0, vmax=1)
            _heatmap(axs_target[1], self.weights, x_range=wl_range, y_range=angle_range, vmin=0)
            axs_target[0].set_title('Target')
            axs_target[1].set_title('Weights')
            for ax in axs_target:
                ax.set_xlabel('Wavelength [nm]')
                ax.set_ylabel('Angle [deg, °]')

        plt.show(block=False)
        plt.pause(0.1)
        return [f_target, axs_target]

    def simulate(self, n, d):
        """
        Simulates the optical response of one cladded stack, averaged over s and p polarization.

        Parameters:
        -----------
        n : np.array of shape [(Sub + L + Am) x S]
            Refractive indices of substrate, stacked layers and ambient
        d : np.array of shape [Sub + L + Am]
            The corresponding thicknesses in meter

        Returns:
        --------
        np.array of shape [D x S]
            Reflectivity, or transmissivity if mode says so, per angle and wavelength
        """
        result_dicts = tmm('s', n, d, (np.pi/180)*self.angle, self.wl)
        result_dictp = tmm('p', n, d, (np.pi/180)*self.angle, self.wl)
        if self.mode == 'reflectivity':
            rs = result_dicts['R']
            rp = result_dictp['R']
            r = (rs + rp) / 2
            return r
        else:
            ts = result_dicts['T']
            tp = result_dictp['T']
            t = (ts + tp) / 2
            return t

    def create_action(self, mat_number, thickness, is_normalized=True):
        """
        Builds an action for step() from a material index and a thickness, the latter either
        normalized to [0, 1] or given in meter.
        """
        if not is_normalized:
            normalized_thickness = (thickness - self.min_thickness) / (self.max_thickness - self.min_thickness)
        else:
            normalized_thickness = thickness
        action = tuple((mat_number, np.array([normalized_thickness])))
        return action

    def create_stack(self, material_list, thickness_list=None):
        """
        Builds refractive indices and thicknesses for a list of materials, e.g. to define a
        substrate or an ambient for set_cladding().

        Parameters:
        -----------
        material_list : list of int
            Material indices, one-based as in the actions passed to step()
        thickness_list : list of float or None
            The corresponding thicknesses in meter, semi-infinite if left out

        Returns:
        --------
        n : np.array of shape [len(material_list) x S]
        d : np.array of shape [len(material_list)]
        dict
            {'n': n, 'd': d}, ready to hand to set_cladding()
        """
        if thickness_list is not None:
            t = np.stack((thickness_list))
        else:
            # without thicknesses the layers are semi-infinite, as the default cladding is
            t = np.full(len(material_list), np.inf)
        n = []
        for material in material_list:
            n.append(self.N[material-1, :])
        n = np.vstack((n))
        dictionary = {'n': n, 'd': t}
        return n, t, dictionary

    def stack_layers(self, d_array=None, n_array=None):
        """
        Clads a stack with the substrate and ambient set by set_cladding().

        Parameters:
        -----------
        d_array, n_array : array_like or None
            A stack to clad instead of the one currently held by the environment

        Returns:
        --------
        cladded_n : np.array of shape [(Sub + L + Am) x S]
        cladded_d : np.array of shape [Sub + L + Am]
        """

        if n_array is not None:
            n_list = list(n_array)
        else:
            n_list = self.n
        if d_array is not None:
            d_list = list(d_array)
        else:
            d_list = self.d

        if len(n_list) != 0:
            cladded_n = np.vstack((n_list))
            cladded_d = np.vstack((d_list)).reshape(-1, 1)
            cladded_n = np.vstack((self.n_substrate, cladded_n))
            cladded_d = np.vstack((self.d_substrate, cladded_d))
            cladded_n = np.vstack((cladded_n, self.n_ambient))
            cladded_d = np.vstack((cladded_d, self.d_ambient))
        else:
            cladded_n = np.vstack((self.n_substrate, self.n_ambient))
            cladded_d = np.vstack((self.d_substrate, self.d_ambient))
        return cladded_n.squeeze(), cladded_d.squeeze()

    def steps_made(self):
        """Number of layers stacked so far; a step that ended the episode adds none."""
        return len(self.layers)

    def reset_reward_track(self):
        """Clears the tracked baseline errors behind baseline_mse."""
        self._reward_track = []

    def set_initial_layers(self, nmb_of_initial_layers):
        """
        Sets the upper bound on how many random layers reset() stacks before handing the
        environment to the agent; it draws between one and this many.

        Raises:
        -------
        ValueError
            If more initial layers than maximum_layers are requested
        """
        if nmb_of_initial_layers > self.maximum_layers:
            raise ValueError("Initial number of layers already exceeds total number of allowed layers!")
        self._initial_nmb_layers = nmb_of_initial_layers

    @property
    def baseline_mse(self):
        """
        Reference error that the reward is normalized against, see reward_func(). Currently a
        constant 0.4; the commented out alternative averages the tracked errors instead.
        """
        if len(self._reward_track) == 0:
            return 0.4
        else:
            return 0.4  # np.mean(self._reward_track)

    @property
    def num_layers(self)->float:
        """
        Number of layers in the stack, counting a run of consecutive layers of the same material
        as one because they are physically indistinguishable.
        """
        if len(self.layers) == 0:
            return 0
        else:
            counter = 0
            prev_layer = -1
            for layer in self.layers:
                if not layer == prev_layer:
                    counter += 1
                    prev_layer = layer
            return counter

    @staticmethod
    def reward_func(reflectivity, target, weights=None, baseline_mse=1.0, normalization=False, low_reward=0.01, high_reward=1.0):
        """
        Rates an optical response against the target.

        Parameters:
        -----------
        reflectivity : np.array of shape [D x S]
            The simulated response; the name is historical, it holds transmissivity in that mode
        target : np.array of shape [D x S]
            The desired response
        weights : np.array of shape [D x S] or None
            Relative importance of each pixel, where zero drops a pixel from the mean
        baseline_mse : float
            The error that maps onto low_reward when normalization is set
        normalization : bool
            Whether to map the error exponentially onto [low_reward, high_reward] instead of
            taking exp(-error)
        low_reward, high_reward : float
            Reward at baseline_mse and at zero error respectively

        Returns:
        --------
        reward : float
        baseline_error : float
            Weighted mean absolute deviation from the target
        """
        if weights is None:
            weights = np.ones_like(target)
        else:
            assert np.all(weights >= 0), 'weights are supposed to be non-negative!'
        temp = np.abs(reflectivity - target) * weights
        temp[weights == 0] = np.nan
        baseline_error = np.nanmean(temp)
        if normalization:
            assert low_reward > 0, 'low_rewards needs to be non-negative!'
            highest_measureable_reward = high_reward
            lowest_measureable_reward = low_reward  # > 0
            a = highest_measureable_reward
            b = np.log(lowest_measureable_reward / highest_measureable_reward) / baseline_mse
            reward = a * np.exp(b * baseline_error)
        else:
            reward = np.exp(-baseline_error)
        return reward, baseline_error
