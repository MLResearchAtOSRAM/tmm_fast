## tmm_fast

tmm_fast or transfer-matrix-method_fast is a lightweight package to speed up optical planar multilayer thin-film device computation. 
It is essentially build on NumPy and the tmm package from sjbyrnes (https://github.com/sbyrnes321/tmm) but quite a lot faster. 
Depending on the number of layers, wavelenght range and angular range speedups of ~100x are possible.

To complete the package, a dataset generation function using Dask can distribute the computations on all available CPUs to further speed-up
computation for really large amounts of thin-film devices (>1E5) which might be interesting for machine learning applications. 

The physics behind the transfer matrix method can be studied in any textbook on optical devices or in https://arxiv.org/abs/1603.02720
from Steven J. Byrnes

