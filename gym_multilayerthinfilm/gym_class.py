import random
import numpy as np
import gym
from gym import spaces
import tmm_fast_core as tmm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

class MultiLayerThinFilm(gym.Env):
    def __init__(self, N, maximum_layers, target, weights=None, normalization=True, sparse_reward=True, substrate=None, ambient=None, max_thickness=250e-9, min_thickness=15e-9, work_path=''):
        """
        Initialize a new environment for multi-layer thin-film (MLTF) optimization.
        Each layer is determined by its (dispersive) refractive index and a thickness.
        Thus, aside from choosing the material of a next layer to place, a reinforcement learning agent
        must also assign a thickness to this layer. This formulation allows to interpret the subsequent stacking of
        layers as a parameterized Markov decision process. (See publication for details)

        Contributors: Heribert Wankerl, Alexander Luce, Maike Stern
        (Feel free to add your name to the list in case you made major contributions or changes)
                Args:

                  N:                  np.array of shape [M x S]
                where M is the number of available materials and S is the number of supporting points of the spectrum
                N holds the (dispersive, complex) refractive indicies of the available materials
            maximum_layers:     integer
                maximum_layers defines the maximum number of layers to stack
            target:             dictionary
                with keys 'target', 'direction', 'spectrum'
                target['direction'] holds the angles [deg, °] under consideration and is of shape D
                target['spectrum'] holds the spectrum [m] under consideration and is of shape S
                target['target'] holds the pixel-wise target reflectivity of a MLTF and is of shape [D x S]
            weights:            np.array of same shape [D x S]
                This array allows to steer the pixels relative influence on the optimization/reward
            normalization:      Boolean
                Determines whether to exponentially transform the reward or not (Look publication for details)
            sparse_reward:      Boolean
                Determines whether to simulate and reward each intermediate stack or only the final stack
            substrate:          dictionary or None
                Holds the (dispersive, complex) refractive indicies of the substrate materials in the rows of
                substrate['n'] which is of shape np.array of shape [Sub x S] . Sub is the number of materials that form
                the substrate. substrate['d'] is of shape Sub and holds the corresponding thicknesses of each layer.
                If substrate is None (default) it is set to vacuum of infinite thickness.
            ambient:            dictionary or None
                Holds the (dispersive, complex) refractive indicies of the ambient materials in the rows of
                ambient['n'] which is of shape np.array of shape [Am x S] . Am is the number of materials that form
                the ambient. ambient['d'] is of shape Am and holds the corresponding thicknesses of each layer.
                If ambient is None (default) it is set to vacuum of infinite thickness.
            max_thickness:      float
                Determines the maximum layer thickness in meter.
            min_thickness:      float
                Determines the minimum layer thickness in meter.
            work_path:          str
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
        self._reward_track = []
        self.layers = []
        self._initial_nmb_layers = 0

        self.n = []
        self.d = []
        self.f = None
        self.axs = None
        # OpenAI gym related settings:
        # action space:
        self.action_space = spaces.Tuple((spaces.Discrete(self.number_of_materials + 1),
                                          spaces.Box(low=0, high=1, shape=(1,))))
        # simulation state space:
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.target.shape[0],), dtype=np.int)
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
        if substrate is not None:
            self.n_substrate = substrate['n']
            self.d_substrate = substrate['d']
        else:
            self.n_substrate = np.ones((1, self.wl.shape[0]))
            self.d_substrate = np.array([np.inf]).squeeze()
            print('--- substrate is set to vacuum of infinite thickness ---')
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
        This method implements the conduction of an action in the environment. Namely, to stack a layer of a
        particular thickness on top of an existing stack.

                Args:

                  action:                  np.array of shape [2]
                action[0] holds the material chosen by the agent as an integer value. Note that 0 leads to
                termination of stacking. action[1] is a float between 0 and 1 and encodes the assigned thickness.

            Rets:
            self.simulation:         np.array of shape [D x S]
                holds the reflectivity for each direction and wavelength of a particular stack in its entries.
            self.n:                  list
                List of np.arrays of shape [1 x S] that holds the refractive indicies of the stacked layers
            self.d:                  list
                List of floats that determine the thicknesses of each stacked layer.
            self.reward:             float
                rates the current stack based on its fullfillment of the target characteristics
            done:                    Boolean
                done-flag that determines whether to end stacking or not.
            []:                      Empty list
                No info list available
                """

        done = False
        if action[0] == 0 or self.steps_made == self.maximum_layers:
            done = True
        else:
            self.layers.append(action[0])
            n_layer = self.N[int(action[0] - 1), :].reshape(1, -1)
            d_layer = (self.max_thickness - self.min_thickness) * action[1] + self.min_thickness
            self.n.append(n_layer)
            self.d.append(d_layer)
        cladded_n, cladded_d = self.stack_layers()
        self.simulation = self.simulate(cladded_n, cladded_d)
        self.reward = 0
        if done or not self.sparse_reward:
            self.reward, mse = self.reward_func(self.simulation, self.target, self.weights, self.baseline_mse, self.normalization)
            if done:
                self._reward_track.append(mse)  # track reward to compute baseline
        return [self.simulation, self.n, self.d], self.reward, done, []

    def reset(self):
        """
        This method implements the reset of the environment to a initial state. However, to ease exploration,
        a defined number of layers can be stacked before handing it to the agent.

                Args:

                  self

            Rets:
            All of this returns are computed based on the initial stack determined by the user
            self.simulation:         np.array of shape [D x S]
                holds the reflectivity for each direction and wavelength of a particular stack in its entries.
            self.reward:             float
                rates the current stack based on its fullfillment of the target characteristics
            self.n:                  list
                List of np.arrays of shape [1 x S] that holds the refractive indicies of the stacked layers
            self.d:                  list
                List of floats that determine the thicknesses of each stacked layer.
            """
        self.layers = []
        self.n = []
        self.d = []
        if self._initial_nmb_layers > 0:
            num_layers = random.randint(1, self._initial_nmb_layers - 1)
            for _ in range(num_layers):
                rnd_material_idx = random.randint(0, self.number_of_materials)
                rnd_material_d = random.uniform(0, 1)
                self.layers.append(rnd_material_idx)
                n_layer = self.N[rnd_material_idx].reshape(1, -1)
                d_layer = (self.max_thickness - self.min_thickness) * rnd_material_d + self.min_thickness
                self.n.append(n_layer)
                self.d.append(d_layer)
        cladded_n, cladded_d = self.stack_layers()
        self.simulation = self.simulate(cladded_n, cladded_d)

        self.reward = 0
        if not self.sparse_reward:
            self.reward, _ = self.reward_func(self.simulation, self.target, self.weights, self.baseline_mse, self.normalization)
        return [self.simulation, self.n, self.d], self.reward, [], []

    def render(self, conduct_simulation=True):
        """
            This method renders the current multi-layer thin film and associated optical response
                    Args:

                      conduct_simulation:   Boolean
                    states whether to conduct the simulation or use the currently stored

                Rets:
                Figure related objects to e.g. envolve them further
                self.f:         Figure
                self.axs:             ndarray of shape (2) holding AxesSubplots

                """
        colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
        cbar = True
        if self.f is None:
            self.f, self.axs = plt.subplots(nrows=1, ncols=3, )
        else:
            if plt.fignum_exists(self.f.number):
                self.axs[0].clear()
                self.axs[1].clear()
                self.axs[2].clear()
                cbar = False
            else:
                self.f, self.axs = plt.subplots(nrows=1, ncols=3, )
        assert self.N.shape[0] <= len(colors), 'Not enough colors to illustrate all materials in N!'
        # plot reflectivity:
        if conduct_simulation:
            cladded_n, cladded_d = self.stack_layers()
            self.simulation = self.simulate(cladded_n, cladded_d)
        self.reward, _ = self.reward_func(self.simulation, self.target, self.weights, self.baseline_mse, self.normalization)
        # drawing:
        assert self.wl.shape[0] > 1 or self.angle.shape[0] > 1, 'No rendering for single wavelenght and single direction!'
        if self.angle.shape[0] == 1:
            xaxis = np.linspace(0, self.wl.shape[0], 10, dtype=int)
            xtickslabels = np.linspace(np.min(self.wl * 10 ** 9), np.max(self.wl * 10 ** 9), 10, dtype=int)
            plt.sca(self.axs[0])
            plt.plot(self.simulation.squeeze())
            plt.xticks(xaxis, xtickslabels)
            plt.xlabel('Wavelength [nm]')
            plt.ylabel('Reflectivity [1]')
            plt.ylim([0, 1.05])
            plt.title('Reflectivity at incidence angle of ' + str(self.angle[0]) + '°\nReward = ' + str(np.round(self.reward, 4)))
        elif self.wl.shape[0] == 1:
            xaxis = np.linspace(0, self.angle.shape[0], 10, dtype=int)
            xtickslabels = np.linspace(np.min(self.angle), np.max(self.angle), 10, dtype=int)
            plt.sca(self.axs[0])
            plt.plot(self.simulation.squeeze())
            plt.xticks(xaxis, xtickslabels)
            plt.xlabel('Angle [deg, °]')
            plt.ylabel('Reflectivity [1]')
            plt.ylim([0, 1.05])
            plt.title('Reflectivity at wavelength ' + str(np.round(self.wl[0] * 10 ** 9, 3)) + ' nm\nReward = ' + str(np.round(self.reward, 4)))
        else:
            yticks = np.linspace(0, self.target.shape[0], 10, dtype=int)
            ytickslabels = np.linspace(np.min(self.angle), np.max(self.angle), 10, dtype=int)
            xticks = np.linspace(0, self.target.shape[1], 10, dtype=int)
            xtickslabels = np.linspace(np.min(self.wl*10**9), np.max(self.wl*10**9), 10, dtype=int)
            colormap = None  # 'twilight'
            g = sns.heatmap(self.simulation, vmin=np.min(self.simulation), vmax=np.max(self.simulation), ax=self.axs[0], xticklabels=xtickslabels, yticklabels=ytickslabels,
                            cmap=colormap, cbar=cbar)
            g.set_xticks(xticks)
            g.set_yticks(yticks)
            g.set_ylabel('Angle [deg, °]')
            g.set_xlabel('Wavelength [nm]')
            g.set_xticklabels(g.get_xticklabels(), rotation=45)
            g.set_yticklabels(g.get_yticklabels(), rotation=0)
            g.set_title(self.mode + '\nReward = ' + str(np.round(self.reward, 4)))

        # plot stack:
        plt.sca(self.axs[1])
        self.axs[1].yaxis.tick_right()
        self.axs[1].yaxis.set_label_position("right")
        self.axs[1].yaxis.grid(linestyle='dotted')
        # for major ticks
        self.axs[1].set_xticks([])
        # for minor ticks
        self.axs[1].set_xticks([], minor=True)
        self.axs[1].ticklabel_format(axis='y', style='sci', scilimits=(-9, -9), useOffset=None, useLocale=None, useMathText=None)
        ind = np.array([1])
        width = 0.25
        for layer, nidx in enumerate(self.layers):
            plt.bar(ind[0], self.d[layer], width, bottom=np.sum(self.d[:layer]), color=colors[nidx-1])
        num_materials = self.num_layers
        plt.xticks(ind,
                   ('Multi-layer thin film\n' + str(num_materials) + ' layers',))
        plt.ylabel('Thickness [m]')

        # # # MATERIAL LEGEND:
        plt.sca(self.axs[2])
        plt.axis('off')
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=colors[idx], lw=10, label='Material ' + str(idx+1)) for idx in range(self.N.shape[0])]
        plt.legend(handles=legend_elements, loc='center right')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        return [self.f, self.axs]

    def render_target(self):
        assert self.wl.shape[0] > 1 or self.angle.shape[0] > 1, 'No rendering for single wavelenght and single direction!'
        f_target, axs_target = plt.subplots(nrows=1, ncols=2, )
        # plot target:
        if self.angle.shape[0] == 1:
            xaxis = np.linspace(0, self.wl.shape[0], 10, dtype=int)
            xtickslabels = np.linspace(np.min(self.wl * 10 ** 9), np.max(self.wl * 10 ** 9), 10, dtype=int)
            #target:
            plt.sca(axs_target[0])
            plt.plot(self.target.squeeze())
            plt.xticks(xaxis, xtickslabels)
            plt.xlabel('Wavelength [nm]')
            plt.ylabel(self.mode + ' [1]')
            plt.ylim([0, 1.05])
            plt.title('Target at incidence angle of ' + str(self.angle[0]) + ' °')
            #weights
            plt.sca(axs_target[1])
            plt.plot(self.weights.squeeze())
            plt.xticks(xaxis, xtickslabels)
            plt.xlabel('Wavelength [nm]')
            plt.ylabel('Weight [1]')
            plt.ylim([0, 1.05 * np.max(self.weights)])
            plt.title('Weights at incidence angle of ' + str(self.angle[0]) + ' °')
        elif self.wl.shape[0] == 1:
            xaxis = np.linspace(0, self.angle.shape[0], 10, dtype=int)
            xtickslabels = np.linspace(np.min(self.angle), np.max(self.angle), 10, dtype=int)
            #target
            plt.sca(axs_target[0])
            plt.plot(self.target.squeeze())
            plt.xticks(xaxis, xtickslabels)
            plt.xlabel('Angle [deg, °]')
            plt.ylabel(self.mode + ' [1]')
            plt.ylim([0, 1.05])
            plt.title('Target at wavelength ' + str(np.round(self.wl[0] * 10 ** 9, 3)) + ' nm')
            # weights
            plt.sca(axs_target[1])
            plt.plot(self.weights.squeeze())
            plt.xticks(xaxis, xtickslabels)
            plt.xlabel('Angle [deg, °]')
            plt.ylabel('Weight [1]')
            plt.ylim([0, 1.05 * np.max(self.weights)])
            plt.title('Weights at wavelength ' + str(np.round(self.wl[0] * 10 ** 9, 3)) + ' nm')
        else:
            yticks = np.linspace(0, self.target.shape[0], 10, dtype=int)
            ytickslabels = np.linspace(np.min(self.angle), np.max(self.angle), 10, dtype=int)
            xticks = np.linspace(0, self.target.shape[1], 10, dtype=int)
            xtickslabels = np.linspace(np.min(self.wl * 10 ** 9), np.max(self.wl * 10 ** 9), 10, dtype=int)
            colormap = None  # 'twilight'
            # target:
            g = sns.heatmap(self.target, vmin=0, vmax=1, ax=axs_target[0], xticklabels=xtickslabels, yticklabels=ytickslabels, cmap=colormap)
            g.set_xticks(xticks)
            g.set_yticks(yticks)
            g.set_ylabel('Angle [deg, °]')
            g.set_xlabel('Wavelength [nm]')
            g.set_xticklabels(g.get_xticklabels(), rotation=45)
            g.set_yticklabels(g.get_yticklabels(), rotation=0)
            g.set_title('Target over angle and spectrum')
            # weights:
            g = sns.heatmap(self.weights, vmin=0, ax=axs_target[1], xticklabels=xtickslabels, yticklabels=ytickslabels, cmap=colormap)
            g.set_xticks(xticks)
            g.set_yticks(yticks)
            g.set_ylabel('Angle [deg, °]')
            g.set_xlabel('Wavelength [nm]')
            g.set_xticklabels(g.get_xticklabels(), rotation=45)
            g.set_yticklabels(g.get_yticklabels(), rotation=0)
            g.set_title('Weights over angle and spectrum')
            plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
        return [f_target, axs_target]

    def simulate(self, n, d):
        """
        This method implements the simulation of the reflectivity of a particular stack. The TMM and its
        parallelization is based on previous work of Alexander Luce.

                Args:

                 n:                     np.array of shape [(Sub + L + Am) x S]
                n holds the refractive indicies of L stacked layers by the agent, including substrate and ambient.
            d:                     np.array of shape Sub + L + Am
                d holds the thicknesses in meter of the layers

            Rets:

            r:                     np.array of shape [D x S]
                r holds the pixel-wise reflectivity values for the directions and wavelengths under consideration
            """
        result_dicts = tmm.coh_tmm_fast_disp('s', n, d, (np.pi/180)*self.angle, self.wl)
        result_dictp = tmm.coh_tmm_fast_disp('p', n, d, (np.pi/180)*self.angle, self.wl)
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

    def create_action(self, mat_number, thickness):
        normalized_thickness = (thickness - self.min_thickness) / (self.max_thickness - self.min_thickness)
        action = tuple((mat_number, np.array([normalized_thickness])))
        return action

    def create_stack(self, material_list, thickness_list=None):
        if thickness_list is not None:
            t = np.stack((thickness_list))
        else:
            t = np.empty()
        n = []
        for material in material_list:
            n.append(self.N[material-1, :].reshape(1, -1))
        n = np.vstack((n))
        dictionary = {'n': n, 'd': t}
        return n, t, dictionary

    def stack_layers(self, d_array=None, n_array=None):
        """
        This method clads the stack suggested by the agent with the pre-defined cladding.
        The returned arrays n, d describe a particular stack, it includes the cladding.
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
        """
        Returns the number of steps made in the environment
            """
        return len(self.layers)

    def reset_reward_track(self):
        """
        To reset private property _reward_track.
            """
        self._reward_track = []

    def set_initial_layers(self, nmb_of_initial_layers):
        """
        Setter for the private property that specifies an initial number of layers during environment reset
            """
        if nmb_of_initial_layers < self.maximum_layers:
            raise ValueError("Initial number of layers already exceeds total number of allowed layers!")
        self._initial_nmb_layers = nmb_of_initial_layers

    @property
    def baseline_mse(self):
        """
        Returns the baseline mse for reward computation/transformation (See publication for details)
            """
        if len(self._reward_track) == 0:
            return 1.0
        else:
            return np.mean(self._reward_track)

    @property
    def num_layers(self):
        """
        Returns the explicit number of layers of a stack
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
        An unconstrained reward computation based on the observed reflectivity and the given target. You can write here whatever you prefer:
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
            reward = - baseline_error
        return reward, baseline_error
