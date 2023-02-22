import math

class Value:

    def __init__(self, data, _parents=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        
        self._prev = set(_parents) # children is transformed from tuple to set for performance reasons
        self._op =  _op
        self._backward = lambda: None # this function decides how gradient will be calculated for backpropagation

        # only for testing and visualization TODO: remove when finished
        self.label = label

    def backward(self):
        topo = []
        visited = set()
        def build_topo(node):
            # topological sorting of a graph
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # converts to Value() if it isnt already
        out = Value(self.data+other.data, (self, other), '+') 
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) # converts to Value() if it isnt already
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
    
    def __gt__(self, other):
        other = other if isinstance(other, Value) else Value(other) # converts to Value() if it isnt already
        return self.data > other.data
    
    def __abs__(self):
        return -self if self.data < 0 else self
    
    def __neg__(self):
        return self * -1
    
    def __rmul__(self, other):
        return  self * other
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        # other = other if isinstance(other, Value) else Value(other) # converts to Value() if it isnt already
        return self + (-other)
    
    def __rsub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        # other = other if isinstance(other, Value) else Value(other) # converts to Value() if it isnt already
        return  self * (other**-1)
    
    def __rtruediv__(self,other):
        return  self * (other**-1)

    def __repr__(self):
        return f'{self.label}: Value(data={self.data}, grad={self.grad})'
