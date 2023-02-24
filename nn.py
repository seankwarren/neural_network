"""
The `neural_networks` module provides classes for building multi-layer perceptron (MLP)
neural networks.

Classes:
- `Neuron`: a class representing a single neuron in a neural network.
- `Layer`: a class representing a layer of neurons in a neural network.
- `MLP`: a class representing a multi-layer perceptron neural network.

Each class provides a `parameters` method to retrieve the learnable parameters of the network or
layer, as well as a `__call__` method for forward propagation through the network. The `MLP`
class also provides a `zero_grad` method to reset gradients to zero between iterations during
training.

Example usage:

import neural_networks

# create a 2-layer MLP with 4 input neurons, 8 hidden neurons, and 1 output neuron
mlp = neural_networks.MLP(4, [8, 1])

# forward propagate through the network with some input data
output = mlp([1.0, 2.0, 3.0, 4.0])

# retrieve the learnable parameters of the network
params = mlp.parameters()

# reset gradients to 0 before running backpropagation
mlp.zero_grad()

# backpropagation
mlp.backward()
"""
import random
from value import Value

class Module:

    def zero_grad(self):
        """
        Resets the gradients to 0 for all parameters in the MLP.
        """
        for p in self.parameters():
            # reset gradients to 0 between iterations
            p.grad = 0.0

    def parameters(self):
        return []

class Neuron(Module):
    """
    A single neuron with an arbitrary number of input connections.

    Args:
        nin (int): The number of input connections.

    Attributes:
        w (List[Value]): A list of input weights.
        b (Value): The bias weight.

    """
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        """
        Compute the output of the neuron given an input.

        Args:
            x (List[Value]): A list of input values.

        Returns:
            Value: The output of the neuron.

        """
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        """
        Get a list containing the weights and bias of the neuron.

        Returns:
            List[Value]: A list of the weights and biases of the neuron.

        """
        return self.w + [self.b]
    
    def __repr__(self):
        return f"Neuron({len(self.w)})"

class Layer(Module):
    """
    A layer of neurons.

    Args:
        nin (int): The number of input connections for each neuron.
        nout (int): The number of neurons in the layer.

    Attributes:
        neurons (List[Neuron]): A list of neurons in the layer.

    """
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        """
        Compute the output of the layer given an input.

        Args:
            x (List[Value]): A list of input values.

        Returns:
            List[Value] or Value: The output of the layer.

        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        """
        Get a list of the weights and biases of all the neurons in the layer.

        Returns:
            List[Value]: A list of the weights and biases of all the neurons in the layer.

        """
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __iter__(self):
        self._i = 0
        return iter(self.neurons)

    def __next__(self):
        return self.neurons[self._i]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    """
    A multilayer perceptron.

    Args:
        nin (int): The number of inputs to the MLP.
        nouts (List[int]): A list of integers indicating the number of neurons in each layer.

    Attributes:
        layers (List[Layer]): A list of layers in the MLP.
    """

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        """
        Compute the output of the MLP given an input.

        Args:
            x (List[Value]): A list of input values.

        Returns:
            List[Value] or Value: The output of the MLP.

        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """
        Get a list of the weights and biases of all the neurons in the MLP.

        Returns:
            List[Value]: A list of the weights and biases of all the neurons in the MLP.

        """
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __iter__(self):
        self._i = 0
        return iter(self.layers)

    def __next__(self):
        return self.layers[self._i]
    
    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
