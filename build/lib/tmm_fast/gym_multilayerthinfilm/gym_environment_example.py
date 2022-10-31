import gym_multilayerthinfilm as mltf
import numpy as np

# Material specifications
# Pathes were material txt-files are stored are gathered in a list
pathAu = 'materials\\nAu.txt'
pathNb2O5 = 'materials\\nNb2O5.txt'
pathSiO2 = 'materials\\nSiO2.txt'

material_path_list = [pathAu, pathNb2O5, pathSiO2]

# Specification of operation mode as well as angular and spectral range of interest

mode = 'reflectivity'  # 'transmittivity' or 'reflectivity'
maximum_number_of_layers = 10
angle_min = 0  # °
angle_max = 90  # °
lambda_min = 400  # nm
lambda_max = 700  # nm

# Definition of the target reflectivity for each angle and wavelength (pixelwise as an array)
target_array = 0.5 * np.ones((int(angle_max - angle_min), int(lambda_max - lambda_min)))


wl = np.linspace(lambda_min, lambda_max, int(lambda_max - lambda_min)) * 1e-9
angle = np.linspace(angle_min, angle_max, int(angle_max - angle_min))
target = {'direction': angle, 'spectrum': wl, 'target': target_array, 'mode': mode}

N = mltf.get_N(material_path_list, lambda_min, lambda_max, points=int(lambda_max - lambda_min), complex_n=True)
N = np.vstack((N, np.ones((1, N.shape[1]))))

# Creation of the environment given the above information
env = mltf.MultiLayerThinFilm(N, maximum_number_of_layers, target)

env.reset()

# By default the multilayer thin film is cladded by vacuum of infinite thickness.
# Well, let's assume that the light injection layer is actually air (which is very close to vacuum),
# but the ambient consists of gold (Au) of infinite thickness. In case that the ambient consisits of more stacked
# materials, we can just expand the list properly!
# The procedure remains the same for the substrate of course.

#ambient:
ambient_material_list = [2]
ambient_thickness_list = [np.inf]
# The create_stack-method forms a dictionary that defines the ambient:
_, _, ambient_dict = env.create_stack(ambient_material_list, ambient_thickness_list)
env.set_cladding(ambient=ambient_dict)
# We might get warned, if the refractive indexes in the ambient are non-real!


# In the following, a particular multilayer thin film is constructed inbetween the aforementioned cladding
# via consecutive layer stacking. Here, the thin film consists of tow layers (Nb2O5 and SiO2) of thicknesses 10 nm each:
layer_material_list = [2, 3]
for layer in layer_material_list:
    # Execute steps to form multilayer thin film via layer stacking:
    env.step(env.create_action(layer, thickness=10e-9))
# And another three random layers:
env.step(env.action_space.sample())
env.step(env.action_space.sample())
# For the last step, we read out the state-list as well as the reward and done-flag.
# Note that the read-out reward is only non-zero if done==True in case that env.sparse_reward is True, therefore we set:
env.sparse_reward = False
[simulation, n, d], reward, done, _ = env.step(env.action_space.sample())

# Finally, we render the target that was provided above as well as the current state of the environment
# Namely, the state is defined by the currently formed thin film as well as the corresponding optical behavior
env.render_target()
env.render()
print(':)')
