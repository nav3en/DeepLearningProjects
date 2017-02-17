import numpy as np

class Node:
    def __init__(self, inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []

        # The current node becomes the outbound node for the incoming nodes
        for node in self.inbound_nodes:
            node.outbound_nodes.append(self)
        # Each node will calculate values which will be propagated in the n/w
        self.value = None

    # Forward pass method to calculate
    def forward(self):
        raise NotImplemented

# This is a subclass of Node
class Input(Node):
    def __init__(self):
        # Call the superclass constructor
        # Since there are no inbound nodes for the input node we don't pass inbound nodes
        Node.__init__(self)
    def forward(self,value=None):
        if value is not None:
            self.value = value

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
