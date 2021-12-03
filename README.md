
# tmm_fast

tmm_fast or transfer-matrix-method_fast is a lightweight package to speed up optical planar multilayer thin-film device computation. 
Typically, one is interested in computing the Reflection and/or Transmission of light through a multilayer thin-film depending on the 
wavelength and angle of incidence of the incoming light. The package is build on the original tmm package from sjbyrnes 
(https://github.com/sbyrnes321/tmm) but quite a lot faster. Depending on the number of layers, wavelength range and angular range 
speed-ups of ~100x are possible. The physics behind the transfer matrix method can be studied in any textbook on optical devices or in https://arxiv.org/abs/1603.02720
from Steven J. Byrnes.
More detailed informations about the package and its applications can be found at https://arxiv.org/abs/2111.13667

![Alt text](./misc/tmm_structure.svg)

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

tmm_fast requires some prerequisits:
* Numpy
* tmm
* PyTorch >= 1.9
* Dask
* Matplotlib
* Seaborn 
* Gym
  ```sh
  pip install numpy tmm pytorch>=1.9.1 dask
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/MLResearchAtOSRAM/tmm_fast.git
   ```

In case any dependency is not fullfilled, you can create an environment using gym_multilayerthinfilm.yml which is located in the package folder; dont forget to specify your pyhton environment folder/path there (prefix).<br/>
In general, there are no weird dependencies aside from numpy, matplotlib, seaborn, dask and gym. The tmm package can be downloaded/installed from here if necessary:<br/>
pip install git+https://github.com/sbyrnes321/tmm.git 

## tmm_fast_core

tmm_fast_core contains the core functionality of the transfer matrix method, implemented in Numpy. coh_tmm_fast and coh_tmm_fast_disp can be 
used to compute a multilayer thin film. By using the multithread_coh_tmm method, large amounts of thin-films can be computed in parallel by 
using Dask.

## tmm_fast_torch

tmm_fast_torch is a reimplemented version of the tmm_fast_core code. It provides the same functionality as the core methods but allows to 
compute gradients via PyTroch Autograd. In future versions, GPU acceleration and 2nd order gradients could be implemented, too. 

## gym-multilayerthinfilm

## Overview
The proposed OpenAI gym environment utilizes the parallelized transfer-matrix method (TMM-Fast) to implement the optimization of multi-layer thin films as parameterized Markov decision processes. An very intuitve example is provided in example.py.
Whereas the contained physical methods are well-studied and known since decades, the contribution of this code lies the transfer to an OpenAI gym environment. The intention is to enable AI researchers without optical expertise to solve the corresponding parameterized Markov decision processes. Due to their structure, the solution of such problems is still an active field of research in the AI community.<br/>
The publication [Parameterized Reinforcement learning for Optical System Optimization](https://iopscience.iop.org/article/10.1088/1361-6463/abfddb) used this environment.

<!-- ## Installation
1.<br/>
pip install git+https://github.com/MLResearchAtOSRAM/gym-multilayerthinfilm.git<br/><br/>
2.<br/>
Clone the repository and executing setup.py
-->


## Getting started
To get started you can do the tutorial notebook example.ipynb or just check out the quickstarter.py!

## Multi-layer thin films meet parameterized reinforcement learning
Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of reward. The code to be published implements such an environment for the optimization of multi-layer thin films.
In principle, the proposed code allows to execute actions taken by an agent. These actions determine which material of which thickness to stack next, thereby consecutively forming a multi-layer thin film as illustrated in figure 1. Such a multi-layer thin film exhibits optical characteristics. By comparison between the actual and user-defined desired characteristics, a notion of numeric reward is computed based on which the agent learns to distinguish between good and bad design choices. Due to its physical and mathematical structure, the optimization of multi-layer thin film remains a challenging and thus still active field of research in the scientific community. As such it gained recent attention in many publications. Therefore, naturally the need for a standardised environment arises to make the corresponding research more trustful, comparable and consistent.

![image](https://user-images.githubusercontent.com/83709614/127179171-bc7e8fe5-bd83-4125-a84f-12a9e16c3150.png)<br/> 
Figure 1: Principal idea of an OpenAI gym environment. The agent takes an action that specifies the material and thickness of the layer to stack next. The environment implements the multi-layer thin film generation as consecutive conduction of actions and assigns a reward to a proposed multi-layer thin film based on how close the actual (solid orange line) fulfils a desired (dashed orange line) characteristic. The made experience is used to adapt the taken actions made in order to increase the reward and thus generate more and more sophisticated multi-layer thin films.

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

![image](https://user-images.githubusercontent.com/83709614/127179200-16aaf611-ad17-4082-a47f-d933ba7cbc83.png)<br/> 
Figure 2: Rendered output of the environment. Reflectivity (left) over angle of incidence and spectrum of a multi-layer thin film (right). Here, the stack features four layers and each layer’s material was chosen from a set of eight alternatives. The reward is computed based on a desired reflectivity, which is one for each angle and wavelength, but not displayed in this figure.


## Citing

If you use the code from this repository for your projects, please cite:
[TMM-Fast: A Transfer Matrix Computation Package for Multilayer Thin-Film Optimization](https://arxiv.org/abs/2111.13667) in your publications.