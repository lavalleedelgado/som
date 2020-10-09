# som
Patrick Lavallee Delgado \
October 2020

## Overview

A self-organizing map re-projects high-dimensional data onto a terrain whose topology reveals clusters among observations. This implementation follows that proposed by Kohonen (1990) as an unsupervised neural network, in which a grid of nodes learn codebook vectors that approximate the observations in the data most attracted to it. Learning is semi-competitive, by which the best matching node brings itself closer to the data while also pulling the nodes in a neighborhood around it towards the same point. The grid demonstrates topological coherence in the long run.

The self-organizing map accepts four parameters. The shape of the grid sets the number of nodes and the length of a codebook vector. The learning rate *alpha* sets the adaptive gain that an observation induces on its best matching node. The neighborhood restraint *sigma* sets the size of the Gaussian kernel centered on the best matching node through which to propagate a response to an observation. The minimum number of iterations sets the length of training, the first half dedicated to burn-in and the second half to decay the learning rate and neighborhood restraint.

While other self-organizing map packages exist that offer more distance metrics and neighborhood kernel functions, e.g. [minisom](https://github.com/JustGlowing/minisom), none has particularly well-edited or well-documented code that allow a beginner to understand the algorithm. This is a personal learning endeavor.

## Implementation

The `som` package offers a similar interface to `sklearn` objects: the `fit()` method trains the map on the data, the `predict()` method transforms the data to its node assignments, and the `fit_predict()` method executes both.

```
from som import SOM

# Initialize the self-organizing map with default parameters.
shape = 10, 10, X.shape[1]
som = SOM(shape)

# Fit the map to the data and transform the data to its node assignments.
Z = som.fit_predict(X)
```

The U-matrix of a self-organizing map gives the normalized average distance of a node from those adjacent to it. It it helpful to think of this as a birds-eye view of the topology learned from the data: values near zero represent "plains", values near one represent "mountains", and moving between extremes requires traversing more terrain. So, data mapped to a higher elevation on the U-matrix is farther away and more isolated from other observations, even those in adjacent nodes.

```
import matplotlib.pyplot as plt
import numpy as np

# Get the U-matrix representation of the map.
# Transpose ensures correspondence between matrix entries and plot coordinates.
U = som.umatrix.T

# Initialize the figure and a discretized colormap.
fig, ax = plt.subplots()
cmap = plt.cm.get_cmap('gray', 5)

# Plot the U-matrix and a colorbar.
ax.pcolormesh(U, cmap=cmap)
mapper = plt.cm.ScalarMappable(cmap=cmap)
fig.colorbar(mapper, ax=ax)

# Plot the data with random jitter to disperse points over the node.
Z += np.random.random((Z.shape[0], 2)) * 0.9 + 0.05
ax.scatter(Z[:, 0], Z[:, 1])

# Move ticks to the center of each node.
ax.set_xticks(np.arange(U.shape[0]) + 0.5)
ax.set_xticklabels([str(i) for i in range(1, U.shape[0] + 1)])
ax.set_yticks(np.arange(U.shape[1]) + 0.5)
ax.set_yticklabels([str(i) for i in range(1, U.shape[1] + 1)])

# Show the figure.
fig.show()
```

## References

Kohonen, Teuvo. 1990. "The Self-organizing Map." *Proceedings of the IEEE* 78, no. 9 (September): 1464-80.
