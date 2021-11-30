
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
  ```sh
  pip install numpy tmm pytorch>=1.9.1 dask
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/MLResearchAtOSRAM/tmm_fast.git
   ```


## tmm_fast_core

tmm_fast_core contains the core functionality of the transfer matrix method, implemented in Numpy. coh_tmm_fast and coh_tmm_fast_disp can be 
used to compute a multilayer thin film. By using the multithread_coh_tmm method, large amounts of thin-films can be computed in parallel by 
using Dask

## tmm_fast_torch

tmm_fast_torch is a reimplemented version of the tmm_fast_core code. It provides the same functionality as the core methods but allows to 
compute gradients via PyTroch Autograd. In future versions, GPU acceleration and 2nd order gradients could be implemented, too. 


## Citing

If you use the code from this repository for your projects, please cite:
# TMM-Fast: A Transfer Matrix Computation Package for Multilayer Thin-Film Optimization
 https://arxiv.org/abs/2111.13667 in your publications.
