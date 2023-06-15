# gym-multilayerthinfilm

## Overview
The proposed OpenAI/Farama-Foundation gymnasium environment utilizes a parallelized transfer-matrix method (TMM) to implement the optimization of for multi-layer thin films as parameterized Markov decision processes. An very intuitve example is provided in example.py.
Whereas the contained physical methods are well-studied and known since decades, the contribution of this code lies the transfer to an OpenAI/Farama-Foundation gymnasium environment. The intention is to enable AI researchers without optical expertise to solve the corresponding parameterized Markov decision processes. Due to their structure, the solution of such problems is still an active field of research in the AI community.<br/>
The publication [Parameterized Reinforcement learning for Optical System Optimization](https://iopscience.iop.org/article/10.1088/1361-6463/abfddb) used this environment.

## Multi-layer thin films meet parameterized reinforcement learning
Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of reward. The code to be published implements such an environment for the optimization of multi-layer thin films.
In principle, the proposed code allows to execute actions taken by an agent. These actions determine which material of which thickness to stack next, thereby consecutively forming a multi-layer thin film as illustrated in figure 1. Such a multi-layer thin film exhibits optical characteristics. By comparison between the actual and user-defined desired characteristics, a notion of numeric reward is computed based on which the agent learns to distinguish between good and bad design choices. Due to its physical and mathematical structure, the optimization of multi-layer thin film remains a challenging and thus still active field of research in the scientific community. As such it gained recent attention in many publications. Therefore, naturally the need for a standardised environment arises to make the corresponding research more trustful, comparable and consistent.


Figure 1: Principal idea of an OpenAI/Farama-Foundation gymnasium environment. The agent takes an action that specifies the material and thickness of the layer to stack next. The environment implements the multi-layer thin film generation as consecutive conduction of actions and assigns a reward to a proposed multi-layer thin film based on how close the actual (solid orange line) fulfils a desired (dashed orange line) characteristic. The made experience is used to adapt the taken actions made in order to increase the reward and thus generate more and more sophisticated multi-layer thin films.

## Describtion of key features
The environment can include<br/> 
•	cladding of the multi-layer thin film (e.g. substrate and ambient materials),<br/>
•	dispersive and dissipative materials,<br/>
•	spectral and angular optical behavior of multi-layer thin films (See figure 2),<br/>
•	… and many more.<br/>

The environment class allows to <br/>
•	conduct so-called parameterized actions (See publication) that define a multi-layer thin film,<br/>
•	evaluate the generated thin film given a desired optical response, and<br/>
•	render the results (See figure 2). <br/>

In general, the comprehensive optimization of multi-layer thin films in regards of optical reponse encompasses <br/>
•	the number of layers (integer),<br/>
•	the thickness of each layer (float),<br/>
•	the material of each layer (categrial, integer).<br/>


Figure 2: Rendered output of the environment. Reflectivity (left) over angle of incidence and spectrum of a multi-layer thin film (right). Here, the stack features four layers and each layer’s material was chosen from a set of eight alternatives. The reward is computed based on a desired reflectivity, which is one for each angle and wavelength, but not displayed in this figure.


## Getting started
Required packages:<br/>
numpy, matplotlib, seaborn, dask, tmm as specified in env_mltf.yml
based on which you can create an approbiate environment via line command<br/>
conda env create -f env_mltf.yml<br/>
Don't  forget to specify your common python environment path (prefix, last line in env_mltf.yml)!

