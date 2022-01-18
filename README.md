# torchTT
Tensor-Train decomposition in pytorch

Tensor-Train decomposition package written only in Python on top of pytorch. Supports GPU acceleration and automatic differentiation.
It also contains routines for solving linear systems in the TTformat and performing adaptive cross approximation.


## Installation

### Requirements
Following requirements are needed:

- `torch>=1.7.0`
- `numpy>=1.18`
- `opt_einsum`

The GPU (if available) version of pytorch is recommended to be installed. Read the [official installation guide](https://pytorch.org/get-started/locally/) for further info.
### Using pip
You can install the package using the `pip` command:

```
pip install git+https://github.com/ion-g-ion/torchTT
```
One can also clone the repository and manually install the package: 

```
git clone https://github.com/ion-g-ion/torchTT
cd torchtt
pip install .
```
### Using conda



## Content


## Tests 



## Examples
Following example scripts ( as well as ipy notebooks) are provided as part of the documentation:

 * [basic_tutorial.py](examples/basic_tutorial.py) / [basic_tutorial.ipynp](examples/basic_tutorial.ipynb): This contains a basic tutorial on decomposing full tensors in the TT format as well as performing rank rounding, slicing. 
 * [basic_linalg.py](examples/basic_linalg.py) / [basic_linalg.ipynp](examples/basic_linalg.ipynb): This tutorial presents all the algebra operations that can be performed in the TT format.
 * [efficient_linalg.py](examples/efficient_linalg.py) / [efficient_linalg.ipynb](examples/efficient_linalg.ipynb): contains the DMRG for fast matves and AMEN for elementwise inversion in the TT format.
 * [automatic_differentiation.py](examples/automatic_differentiation.py) / [automatic_differentiation.ipynp](examples/automatic_differentiation.ipynb)
 * [cross_interpolation.py](examples/cross_interpolation.py) / [cross_interpolation.ipynb](examples/cross_interpolation.ipynb): In this script, the cross interpolation emthod is exemplified.
 * [system_solvers.py](examples/system_solvers.py) / [system_solvers.ipynb](examples/system_solvers.ipynb): This contains the bais ussage of the multilinear solvers.
 * [cuda.py](examples/cuda.py) / [cuda.ipynb](examples/cuda.ipynb): This provides an example on how to use the GPU acceleration.
 
## Author 
Ion Gabriel Ion, e-mail: ion.ion.gabriel@gmail.com
