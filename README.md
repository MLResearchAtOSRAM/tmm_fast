
# tmm_fast

tmm_fast or transfer-matrix-method_fast is a lightweight package to speed up optical planar multilayer thin-film device computation. 
Typically, one is interested in computing the Reflection and/or Transmission of light through a multilayer thin-film depending on the 
wavelength and angle of incidence of the incoming light. The package is build on the original tmm package from sjbyrnes 
(https://github.com/sbyrnes321/tmm) but quite a lot faster. Depending on the number of layers, wavelength range and angular range 
speed-ups of ~100x are possible. The physics behind the transfer matrix method can be studied in any textbook on optical devices or in https://arxiv.org/abs/1603.02720
from Steven J. Byrnes.

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
  pip install numpy tmm pytorch==1.9.1 dask
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```


## tmm_fast_core

The tmm_fast_core 

## tmm_fast_torch

The tmm_fast_torch file 

To complete the package, a dataset generation function using Dask can distribute the computations on all available CPUs to further speed-up
computation for really large amounts of thin-film devices (>1E5) which might be interesting for machine learning applications. 



