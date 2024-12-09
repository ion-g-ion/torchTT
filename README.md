
<p align="center">
<img src="https://github.com/ion-g-ion/torchTT/blob/main/logo.png?raw=true" width="400px" >
</p>

# torchTT
Tensor-Train decomposition in `pytorch`

Tensor-Train decomposition package written in Python on top of `pytorch`. Supports GPU acceleration and automatic differentiation.
It also contains routines for solving linear systems in the TT format and performing adaptive cross approximation  (the AMEN solver/cross interpolation is inspired form the [MATLAB TT-Toolbox](https://github.com/oseledets/TT-Toolbox)).
Some routines are implemented in C++ for an increased execution speed.


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
pip install torchTT
```

The latest github version can be installed using:

```
pip install git+https://github.com/ion-g-ion/torchTT
```

One can also clone the repository and manually install the package: 

```
git clone https://github.com/ion-g-ion/torchTT
cd torchTT
python setup.py install
``` 

### Using conda

**TODO**

## Components

The main modules/submodules that can be accessed after importing `torchtt` are briefly desctibed in the following table.
Detailed description can be found [here](https://ion-g-ion.github.io/torchTT/index.html).

| Component | Description |
| --- | --- |
| [`torchtt`](https://ion-g-ion.github.io/torchTT/torchtt/torchtt.html)             | Basic TT class and basic linear algebra functions. |
| [`torchtt.solvers`](https://ion-g-ion.github.io/torchTT/torchtt/solvers.html)     | Implementation of the AMEN solver. |
| [`torchtt.grad`](https://ion-g-ion.github.io/torchTT/torchtt/grad.html)        | Wrapper for automatic differentiation. |
| [`torchtt.manifold`](https://ion-g-ion.github.io/torchTT/torchtt/manifold.html)    | Riemannian gradient and projection onto manifolds of tensors with fixed TT rank. |
| [`torchtt.nn`](https://ion-g-ion.github.io/torchTT/torchtt/nn.html)          | Basic TT neural network layer. |
| [`torchtt.interpolate`](https://ion-g-ion.github.io/torchTT/torchtt/interpolate.html) | Cross approximation routines. |

## Tests 

The directory [tests/](tests/) from the root folder contains all the `unittests`. To run them use the command:

```
pytest tests/
```


## Documentation and examples
The documentation can be found [here](https://ion-g-ion.github.io/torchTT/index.html).
Following example scripts (as well as python notebooks) are also provied provided as part of the documentation:

 * [basic_tutorial.py](examples/basic_tutorial.py) / [basic_tutorial.ipynp](examples/basic_tutorial.ipynb): This contains a basic tutorial on decomposing full tensors in the TT format as well as performing rank rounding, slicing ([Try on Google Colab](https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/examples/basic_tutorial.ipynb)). 
 * [basic_linalg.py](examples/basic_linalg.py) / [basic_linalg.ipynp](examples/basic_linalg.ipynb): This tutorial presents all the algebra operations that can be performed in the TT format ([Try on Google Colab](https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/examples/basic_linalg.ipynb)). 
 * [efficient_linalg.py](examples/efficient_linalg.py) / [efficient_linalg.ipynb](examples/efficient_linalg.ipynb): contains the DMRG for fast matves and AMEN for elementwise inversion in the TT format ([Try on Google Colab](https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/examples/efficient_linalg.ipynb)). 
 * [automatic_differentiation.py](examples/automatic_differentiation.py) / [automatic_differentiation.ipynp](examples/automatic_differentiation.ipynb): Basic tutorial on AD in `torchtt` ([Try on Google Colab](https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/examples/automatic_differentiation.ipynb)). 
 * [cross_interpolation.py](examples/cross_interpolation.py) / [cross_interpolation.ipynb](examples/cross_interpolation.ipynb): In this script, the cross interpolation emthod is exemplified ([Try on Google Colab](https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/examples/cross_interpolation.ipynb)). 
 * [system_solvers.py](examples/system_solvers.py) / [system_solvers.ipynb](examples/system_solvers.ipynb): This contains the bais ussage of the multilinear solvers ([Try on Google Colab](https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/examples/system_solvers.ipynb)). 
 * [cuda.py](examples/cuda.py) / [cuda.ipynb](examples/cuda.ipynb): This provides an example on how to use the GPU acceleration ([Try on Google Colab](https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/examples/cuda.ipynb)). 
 * [basic_nn.py](examples/basic_nn.py) / [basic_nn.ipynb](examples/basic_nn.ipynb): This provides an example on how to use the TT neural network layers ([Try on Google Colab](https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/examples/basic_nn.ipynb)). 
 * [mnist_nn.py](examples/mnist_nn.py) / [mnist_nn.ipynb](examples/mnist_nn.ipynb): Example of TT layers used for image classification ([Try on Google Colab](https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/examples/mnist_nn.ipynb)). 
 
 The documentation is generated using `shpinx` with:

 ```
 make html
 ```

 after installing the packages

 ```
 pip install sphinx sphinx_rtd_theme
 ```

## Author 
Ion Gabriel Ion, e-mail: ion.ion.gabriel@gmail.com
