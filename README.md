## tmm_fast: Sponge PyTorch functionality for free
Parallelized computation of reflection and transmission for coherent light spectra that traverse
a bunch of multilayer thin-films with dispersive materials.
This package is essentially build on Pytorch and its related functionalities such as GPU accelerations and Autograd.
It naturally allows for:
 - GPU accelerated computations
 - To compute gradients regarding the multilayer thin-film (i.e. N, T) thanks to Pytorch Autograd

In general, tmm_fast is a lightweight package to speed up the computations of reflection and transmission of optical planar multilayer thin-films by vectorization regarding
- different multilayer thin-films
- a set of wavelengths
- a set of incident angles
 
## For old guards: Numpy is fully supported
However, the input can also be a numpy array format.
Although all internal computations are processed via PyTorch, the output data is converted to numpy arrays again.
Hence, the use of numpy input may increase computation time due to data type conversions.


## Benefits and conducted sanity checks, backgrounds
Depending on the number of thin films an their number of layers as well as the considered wavelengths that irradiate the thin film under particular angles of incident, the computation time can be decreased by 2-3 orders of magnitude.
This claim is supported by several cross-checks (https://arxiv.org/abs/2111.13667), conducted with the code provided by Steven J. Byrnes (https://arxiv.org/abs/1603.02720). Of course, the checks covered both, computational time and physical outputs.

The physics behind the transfer matrix method can be studied in any textbook on optical devices or related papers, e.g.
- Chapter on thin films in Microphotonics (http://www.photonics.intec.ugent.be/download/ocs129.pdf) by Dries Van Thourhout, Roel Baets (UGent) and Heidi Ottevaere (VUB)
- Multilayer optical computations (https://arxiv.org/abs/1603.02720) by Steven J. Byrnes
- The Fresnel Coefficient of Thin Film Multilayer Using Transfer Matrix Method (https://iopscience.iop.org/article/10.1088/1757-899X/518/3/032026) by Zahraa Hummam Mohammed
