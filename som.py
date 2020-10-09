from __future__ import annotations
from typing import Tuple, List, Set, Dict, Callable, Iterator, Any
import warnings
from itertools import product
import numpy as np


# Identify default initialization parameters.
DEFAULT_ALPHA = 0.9
DEFAULT_SIGMA = 1.0
DEFAULT_ITERS = 10000


class SOM:
    '''
    Implement the self-organizing map proposed by Kohonen (1990). This is an
    unsupervised neural network that re-projects high dimensional data onto a
    terrain, represented by a grid of nodes whose codebook vectors approximate
    the observations most attracted to it. Training pulls the best matching node
    and those in its neighborhood closer to the data. The resulting topology
    reveals clusters in the data.
    
    Algorithm:
        1. Initialize a grid of nodes with random weights.
        2. Select an observation at random.
        3. Calculate its distance to each node in the grid.
        4. Select the node with the shortest distance as the best-matching unit.
        5. Calculate a neighborhood kernel centered on the best-matching unit.
        6. Update the grid with the product of the kernel and shortest distance.
        7. Repeat from step no. 2.

    Parameters:
        1. shape: number of nodes in x, y and length of a codebook vector d.
        2. alpha: learning rate that decays over time.
        3. sigma: neighborhood restraint the decays over time.
        4. iters: number of iterations in training.

    Notes:
        1. First half of iterations in training are burn-in.
        2. Learning rate and neighborhood restraint do not decay during burn-in.
        3. Training data are row vectors.
    '''

    def __init__(
        self,
        shape: Tuple[int, int, int],
        alpha: float = DEFAULT_ALPHA,
        sigma: float = DEFAULT_SIGMA,
        iters: int = DEFAULT_ITERS
    ) -> None:
        '''
        Initialize a self-organizing map.

        shape (tuple): grid size and dimension of the weights as (x, y, d).
        alpha (float): learning rate on the interval [0, 1).
        sigma (float): neighborhood restraint.
        iters (int): number of iterations in which to fit the map.
        '''
        # Save the initialization parameters.
        self.shape = shape
        self.alpha = alpha
        self.sigma = sigma
        # Set the number of iterations for burn-in and learning.
        self.__set_iterations(iters)
        # Set placeholders that assist calculating the Gaussian kernel.
        self.__gx = np.arange(shape[0])
        self.__gy = np.arange(shape[1])
        # Initialize weights with random values on the interval [-1, 1].
        self.weights = np.random.random(shape) * 2 - 1
        # Initialize a placeholder for the U-matrix.
        self.umatrix = np.zeros(shape[:2])


    def __set_iterations(
        self,
        iters: int
    ) -> None:
        '''
        Set the number of iterations for burn-in and training from the total
        iterations requested.

        iters (int): number of iterations.
        '''
        # Round up the number of iterations to the next thousand.
        self.iters = (iters // 1000 + 1) * 1000
        # Set burn-in to half of the number of iterations.
        self.__iters_burn = self.iters // 2
        # Set learning to the iterations that remain.
        self.__iters_learn = self.iters - self.__iters_burn


    def __verify_iterations(
        self,
        n: int
    ) -> bool:
        '''
        Verify whether there are enough iterations to possibly visit each
        observation in the data during learning at least once.

        n (int): number of observations in the data.

        Return truth value (bool).
        '''
        return n < self.__iters_learn


    def __verify_dimension(
        self,
        d: int
    ) -> bool:
        '''
        Verify whether the dimension of the data is that of the nodes.

        d (int): dimension of the data.

        Return truth value (bool).
        '''
        return self.shape[-1] == d


    def __get_distances(
        self,
        x: np.ndarray
    ) -> float:
        '''
        Calculate the Euclidean distance of some data from each node in the map.

        x (np.ndarray): data with shape (d, 1).

        Return distance matrix with shape (x, y, d) (np.ndarray).
        '''
        return np.linalg.norm(self.weights - x, axis=2)


    def __get_bmu(
        self,
        x: np.ndarray
    ) -> Tuple[int, int]:
        '''
        Get the coordinates of the node that corresponds to the best-matching
        unit, that with the smallest entry in the distance matrix.

        x (np.ndarray): data with shape (d, 1).

        Return x, y coordinates of the BMU in the map (int, int).
        '''
        # Calculate the distance from each node to this observation.
        D = self.__get_distances(x)
        # Return the best-matching unit.
        return np.unravel_index(D.argmin(), D.shape)


    def __get_decay(
        self,
        r: float,
        i: int
    ) -> float:
        '''
        Calculate the exponential decay of a parameter per the current iteration
        during learning. Return the original parameter during burn-in time.

        r (float): parameter to decay.
        i (int): current iteration.

        Return parameter with decay (float).
        '''
        # Return the parameter if the iteration corresponds to burn-in.
        if i < self.__iters_burn:
            return r
        # Decay the parameter for learning otherwise.
        C = self.__iters_learn / 100
        return (C * r) / (C + i % self.__iters_learn)


    def __get_kernel(
        self,
        u: Tuple[int, int],
        sigma: float
    ) -> np.ndarray:
        '''
        Calculate the neighborhood centered around a node u and moderated by a
        restraint sigma as a Gaussian kernel with shape (x, y) of rates by which
        to propagate an update to each node in the map.

        u (int, int): x, y coordinates of a node in the map.
        sigma (float): neighborhood restraint on the Gaussian around u.

        Return kernel (np.ndarray).
        '''
        # Calculate the denominator in the exponent.
        d = 2 * sigma ** 2
        # Separate the kernel into Gaussians in x and y.
        gx = np.exp(-np.power(self.__gx - u[0], 2) / d)
        gy = np.exp(-np.power(self.__gy - u[1], 2) / d)
        # Construct the kernel as the outer product of these Gaussians.
        return np.outer(gx, gy)


    def __set_update(
        self,
        N: np.ndarray,
        x: np.ndarray
    ) -> None:
        '''
        Propagate an update to each node in the map per a neighborhood kernel
        centered around the best-matching unit.

        N (np.ndarray): kernel centered around the BMU with shape (x, y).
        x (np.ndarray): data with shape (d, 1).
        '''
        # Increment nodes by their corresponding updates.
        # Multiply i,jth entry in A by the i,jth vector in B.
        self.weights += np.einsum('ij, ijk->ijk', N, x - self.weights)


    def __verify_ordering(
        self
    ) -> bool:
        '''
        Verify that the map achieves topological order. Start with the node at
        the first position in the grid and check that the cumulative distance
        between it and each other in its row or column monotonically increases
        Repeat down the diagonal of the grid.

        Return truth value (bool).
        '''
        # Consider each node on the diagonal of the map.
        for i in range(min(self.shape)):
            # Get this node.
            n = self.weights[i, i]
            # Initialize the last distance with that from self.
            d_last = 0.0
            # Consider each node in this row.
            for j in range(i, self.shape[0]):
                # Calculate the cumulative distance from node i, i to node i, j.
                d = np.linalg.norm(n - self.weights[i, i:j].sum(axis=0))
                # Check whether this is farther than that previously measured.
                if d + d_last < d_last:
                    return False
                # Update the last distance.
                d_last += d
            # Reset the last distance with that from self.
            d_last = 0.0
            # Consider each node in this column.
            for j in range(i, self.shape[1]):
                # Calculate the cumulative distance from node i, i to node j, i.
                d = np.linalg.norm(n - self.weights[i:j, i].sum(axis=0))
                # Check whether this is farther than that previously measured.
                if d + d_last < d_last:
                    return False
                # Update the last distance.
                d_last += d
        # The map achieves topological order.
        return True


    def __set_umatrix(
        self
    ) -> None:
        '''
        Update the U-matrix.
        '''
        # Clear the U-matrix.
        self.umatrix -= self.umatrix
        # Consider the coordinates of each node.
        XY = product(range(self.shape[0]), range(self.shape[1]))
        for x, y in XY:
            # Consider the coordinates of each adjacent node.
            I = range(max(x - 1, 0), min(x + 2, self.shape[0]))
            J = range(max(y - 1, 0), min(y + 2, self.shape[1]))
            for i, j in product(I, J):
                # Calculate the distance from node x, y to node i, j.
                d = np.linalg.norm(self.weights[x, y] - self.weights[i, j])
                # Increment the entry for node x, y by this distance.
                self.umatrix[x, y] += d
        # Normalize the U-matrix.
        self.umatrix /= self.umatrix.max()


    def fit(
        self,
        data: np.ndarray
    ) -> SOM:
        '''
        Fit the map to observations the data.

        data (np.ndarray): observations with shape (n, d).
        '''
        # Get the shape of the data.
        n, d = data.shape
        # Ensure compatible data.
        assert self.__verify_dimension(d)
        # Update the number of iterations if the length of the data requires.
        if not self.__verify_iterations(n):
            self.__set_iterations(n * 2)
        # Consider random draws of the data with replacement.
        for i, idx in enumerate(np.random.choice(n, self.iters)):
            # Get the best-matching unit for this observation.
            u = self.__get_bmu(data[idx])
            # Decay learning rate and neighborhood restraint.
            alpha = self.__get_decay(self.alpha, i)
            sigma = self.__get_decay(self.sigma, i)
            # Calculate the neighborhood around the BMU with the learning rate.
            N = self.__get_kernel(u, sigma) * alpha
            # Update nodes with the distance between the data and the BMU.
            self.__set_update(N, data[idx])
        # Ensure topological ordering after training.
        if not self.__verify_ordering():
            warnings.warn('Training did not preserve topological ordering!')
        # Update the U-matrix.
        self.__set_umatrix()
        # Return self.
        return self


    def predict(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        '''
        Get the best-matching units in the map for each observation in the data.
        
        data (np.ndarray): observations with shape (n, d).

        Return coordinates in x, y (np.ndarray, np.ndarray).
        '''
        # Ensure compatible data.
        assert self.__verify_dimension(data.shape[-1])
        # Get the number of observations in the data.
        n = len(data)
        # Initialize a container for BMU indices.
        labels = np.full((n, 2), -1, dtype=int)
        # Consider each observation in the data.
        for idx in range(n):
            # Get the best-matching unit for this observation.
            labels[idx] = self.__get_bmu(data[idx])
        # Return the labels as coordinates in x, y.
        return labels


    def fit_predict(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        return self.fit(data).predict(data)


    @property
    def nodes(
        self
    ) -> np.ndarray:
        '''
        Get the nodes in the map by their order in the grid.

        Return nodes with shape (x * y, d) (np.ndarray).
        '''
        new_shape = self.shape[0] * self.shape[1], self.shape[2]
        return self.weights.reshape(new_shape)
