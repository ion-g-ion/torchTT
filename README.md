# torchTT
Tensor-Train decomposition in `pytorch`

Tensor-Train decomposition package written only in Python on top of `pytorch`. Supports GPU acceleration and automatic differentiation.
It also contains routines for solving linear systems in the TT format and performing adaptive cross approximation  (the AMEN solver/cross interpolation is inspired form the [MATLAB TT-Toolbox](https://github.com/oseledets/TT-Toolbox)).


## Installation

### Requirements
Following requirements are needed:

- `python>=3.6`
- `torch>=1.7.0`
- `numpy>=1.18`
- [`opt_einsum`](https://pypi.org/project/opt-einsum/)

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

**TODO**

## Components

The main modules/submodules that can be accessed after importing `torchtt` are briefly desctibed in the following table.
Detailed descriptio can be found [here](https://htmlpreview.github.io/?https://github.com/ion-g-ion/torchTT/blob/main/docs/torchtt/index.html).

| Component | Description |
| --- | --- |
| `torchtt`                  | Basic TT class and basic linear algebra functions. |
| `torchtt.solvers`          | Implementation of the AMEN solver. |
| `torchtt.grad`             | Wrapper for automatic differentiation. |
| `torchtt.manifold`         | Riemannian gradient and projection onto manifolds of tensors with fixed TT rank. |
| `torchtt.nn`               | Basic TT neural network layer. |
| `torchtt.interpolate`      | Cross approximation routines. |

## Tests 

The directory [tests/](tests/) from the root folder contains all the `unittests`. To run them use the command:

```
python -m unittest discover tests/
```


## Documentation and examples
The documentation ca be gound [here](https://htmlpreview.github.io/?https://github.com/ion-g-ion/torchTT/blob/main/docs/torchtt/index.html).
Following example scripts (as well as python notebooks) are also provied provided as part of the documentation:

 * [basic_tutorial.py](examples/basic_tutorial.py) / [basic_tutorial.ipynp](examples/basic_tutorial.ipynb): This contains a basic tutorial on decomposing full tensors in the TT format as well as performing rank rounding, slicing. 
 * [basic_linalg.py](examples/basic_linalg.py) / [basic_linalg.ipynp](examples/basic_linalg.ipynb): This tutorial presents all the algebra operations that can be performed in the TT format.
 * [efficient_linalg.py](examples/efficient_linalg.py) / [efficient_linalg.ipynb](examples/efficient_linalg.ipynb): contains the DMRG for fast matves and AMEN for elementwise inversion in the TT format.
 * [automatic_differentiation.py](examples/automatic_differentiation.py) / [automatic_differentiation.ipynp](examples/automatic_differentiation.ipynb): Basic tutorial on AD in `torchtt`.
 * [cross_interpolation.py](examples/cross_interpolation.py) / [cross_interpolation.ipynb](examples/cross_interpolation.ipynb): In this script, the cross interpolation emthod is exemplified.
 * [system_solvers.py](examples/system_solvers.py) / [system_solvers.ipynb](examples/system_solvers.ipynb): This contains the bais ussage of the multilinear solvers.
 * [cuda.py](examples/cuda.py) / [cuda.ipynb](examples/cuda.ipynb): This provides an example on how to use the GPU acceleration.
 
## Author 
Ion Gabriel Ion, e-mail: ion.ion.gabriel@gmail.com
