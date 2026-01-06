"""
Deep TT Density demo

This notebook illustrates how to build and evaluate a `torchtt.nn.TTDensityLayer` on a simple 2D synthetic example: defining Gaussian bases, sampling random parameters, computing the required input size, estimating a Monte-Carlo integral, and visualizing the resulting density over a grid.
"""

import torch
import torchtt
import torchtt.functional
import matplotlib.pyplot as plt


# Intro constants and layer setup
# 
# Define grid sizes, TT ranks, Gaussian bases, and instantiate the `TTDensityLayer`, along with computing the expected flattened input size.

N = [10, 11]
R = [1, 5, 1]
basis = [
    torchtt.functional.GaussianBasis(torch.linspace(0, 1, N[0]), delta_overlap=1),
    torchtt.functional.GaussianBasis(torch.linspace(0, 1, N[1]), delta_overlap=1),
]
layer = torchtt.nn.TTDensityLayer(N, R, basis, linear_transformation=True)

n_in = torchtt.nn.TTDensityLayer.input_requireemnt(N, R, linear_transformation=True)
print(f"Size of the input is: {n_in}")


# Monte-Carlo evaluation
# 
# Sample random TT inputs and spatial coordinates, evaluate the density layer, and estimate the integral via Monte-Carlo averaging.

n_b = 2**18

tts_in = torch.rand(1, n_in).repeat(n_b, 1)
xs_in = torch.rand(n_b, 2) * 3 - 1.5

xs_in += tts_in[:, -2:]
pdf_val = layer(tts_in, xs_in)
integral = torch.sum(pdf_val, dim=0)

print(pdf_val.shape)
print("Integral using MC: ", integral / n_b * 3**2)


# 2D visualization
# 
# Generate a meshgrid, reuse one TT parameter set, evaluate the density over the grid, and contour-plot the resulting values.

# Create meshgrid for the two dimensions
n_grid = 256
x = torch.linspace(-1.5, 1.5, n_grid)
y = torch.linspace(-1.5, 1.5, n_grid)
# indexing='ij' is the default for torch, creating matrix indexing
X, Y = torch.meshgrid(x, y, indexing='ij')

# Stack to create input xs_in (flattened)
xs_plot = torch.stack((X.flatten(), Y.flatten()), dim=1)

# Use the first parameter set from the batch and expand
# We take the first random initialization from tts_in
tts_plot = tts_in[0].unsqueeze(0).expand(xs_plot.shape[0], -1)

# Evaluate the layer
with torch.no_grad():
    pdf_plot = layer(tts_plot, xs_plot)

# Reshape for plotting
pdf_plot = pdf_plot.reshape(n_grid, n_grid)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(X.numpy(), Y.numpy(), pdf_plot.numpy(), levels=50)
plt.colorbar(label='Density')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('2D density')
plt.show()

