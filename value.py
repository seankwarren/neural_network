import math

class Value:
    """Represents a node in a computational graph for automatic differentiation.

    Args:
        data (float or int): The value associated with the node.
        _children (tuple, optional): A tuple of children nodes in the graph. Defaults
                                     to an empty tuple.
        _op (str, optional): A string representing the operation performed on the node.
                             Defaults to an empty string.
        label (str, optional): A label to identify the node. Defaults to an empty string.

    Attributes:x
        data (float or int): The value associated with the node.
        grad (float): The gradient of the node.
        _prev (set): A set of children nodes in the graph.
        _op (str): A string representing the operation performed on the node.
        _backward (function): A function that computes the gradient of the node.

    Methods:
        backward(): Performs a backward pass through the computational graph and computes the
                    gradients of each node using reverse-mode automatic differentiation.
    Example:
        # create a node with a value of 3.0
        x = Value(3.0)

        # create a node with a value of 4.0
        y = Value(4.0)

        # perform addition between the two nodes
        z = x + y

        # compute the gradient of the node z
        z.backward()

        # print the gradient of the node z
        print(z.grad)"""

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        # children is transformed from tuple to set for performance reasons
        self._prev = set(_children)
        self._op =  _op
        self._backward = lambda: None

        self.label = label

    def backward(self):
        """
        Performs a backward pass through the computational graph of a neural network and computes
        the gradients of each node using reverse-mode automatic differentiation.

        Args:
            self: The root node of the computational graph.

        Returns:
            None. The function updates the gradients of each node in the computational graph.

        Example Usage:
            Let's assume that we have a neural network represented by a computational graph, and
            we want to compute the gradients of each node. We can do this by calling the backward()
            method on the root node of the graph as follows:

                MultiLayerPerceptron(inputs).backward()

            This will update the gradients of each node in the graph, which can then be used for
            parameter updates during the optimization process.
        """

        # initialize an empty list to store the sorted nodes
        topo = []
        # initialize a set to keep track of visited nodes
        visited = set()

        # define a helper function to perform topological sorting
        def build_topo(node):
            # topological sorting of a graph
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        # perform topological sorting starting from the root node
        build_topo(self)

        # set the gradient of the output node to 1.0
        self.grad = 1.0

        # compute the gradients of each node in reverse order
        for node in reversed(topo):
            node._backward()

    # Base binary operations
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # converts to Value() if needed
        out = Value(self.data+other.data, (self, other), '+') 

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) # converts to Value() if needed
        out = Value(self.data*other.data, (self, other), '*')

        def _backward():
            self.grad += (other.data * out.grad)
            other.grad += (self.data * out.grad)
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "__pow__ only support int and float exponents"
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    # Unary operations
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out > 0) * out.grad
        out._backward = _backward

        return out

    def __abs__(self):
        return -self if self.data < 0 else self

    def __neg__(self):
        return self * -1

    # More Binary operations
    def __rmul__(self, other):
        return  self * other

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return  self * (other**-1)

    def __rtruediv__(self,other):
        return  self * (other**-1)

    # Binary comparison operations
    def __gt__(self, other):
        other = other if isinstance(other, Value) else Value(other) # converts to Value() if needed
        return self.data > other.data

    def __ge__(self, other):
        other = other if isinstance(other, Value) else Value(other) # converts to Value() if needed
        return self.data >= other.data

    def __lt__(self, other):
        return not (self >= other)

    def __le__(self, other):
        return not ( self > other)

    def __repr__(self):
        return f'{self.label}: Value(data={self.data}, grad={self.grad})'
