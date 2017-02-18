import numpy as np

class Node:
    def __init__(self, inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []


        self.gradients={}
        # The current node becomes the outbound node for the incoming nodes
        for node in self.inbound_nodes:
            node.outbound_nodes.append(self)
        # Each node will calculate values which will be propagated in the n/w
        self.value = None

    # Forward pass method to calculate
    def forward(self):
        raise NotImplemented

    def backward(self):
        raise NotImplemented

# This is a subclass of Node
class Input(Node):
    def __init__(self):
        # Call the superclass constructor
        # Since there are no inbound nodes for the input node we don't pass inbound nodes
        Node.__init__(self)
    def forward(self):
        pass
    def  backward(self):
        self.gradients = {self: 0}

        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1


class Add(Node):
    def __init__(self,*inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value =sum([node.value for node in self.inbound_nodes])

class Mul(Node):
    def __init__(self,*inputs):
        Node.__init__(self, inputs)
    def forward(self):
        self.value = np.prod([node.value for node in self.inbound_nodes])

class Linear(Node):
    def __init__(self,inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

    def forward(self):
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        bias = self.inbound_nodes[2].value

        self.value = np.dot(X,W) + bias
    def backward(self):

        # gradients will  have values to pass back to incoming nodes
        # Initialize with zeros
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]

            # Setting the gradients for the inputs - grad * weights
            self.gradients[self.inbound_nodes[0]] = np.dot(grad_cost, self.inbound_nodes[1].value.T)

            #Setting gradients for the weights - grad * inputs
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)

            #Setting the gradients for the bias - sum of grad_cost
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost,axis=0, keepdims=False)

class Sigmoid(Node):
    def __init__(self, input_node):
        Node.__init__(self, [input_node])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used with `backward` as well.

        `x`: A numpy array-like object.
        """
        return 1. / (1. + np.exp(-x))

    def forward(self):
        x = self.inbound_nodes[0].value
        self.value = self._sigmoid(x)

    def backward(self):
        self.gradients = {n:np.zeros_like(n.value) for n in self.inbound_nodes}

        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            # Gradient for incoming nodes will be - grad_cost * differential of sigmoid function
            self.gradients[self.inbound_nodes[0]] += (self._sigmoid(self.inbound_nodes[0].value)*(1-self._sigmoid(self.inbound_nodes[0].value)))*grad_cost


class MSE(Node):
    def __init__(self,y,a):
        Node.__init__(self, [y,a])

    def forward(self):
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)

        self.m = self.inbound_nodes[0].value.shape[0]
        self.diff = y - a

        self.value = np.mean(self.diff**2)

    def backward(self):
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff



def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value


def forward_and_backward_pass(graph):
    # Forward pass to calcualte output
    for n in graph:
        n.forward()

    # Backward pass to perform back propagation
    # Starting from the output node and iterating backwards through the sorted list of nodes
    for n in graph[::-1]:
        n.backward()

def sgd_update(trainables, learning_rate=1e-2):
    #trainables - list of input nodes rep the weights/ biases

    for t in trainables:
        #Perform the update
        t.value = learning_rate * t.gradients[t]