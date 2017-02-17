from Miniflow import *


X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])


X, W, b = Input(), Input(), Input()

feed_dict = {X: X_, W: W_, b: b_}

f = Linear(X, W, b)
g = Sigmoid(f)


sorted_nodes = topological_sort(feed_dict)
sigmoid_ouput= forward_pass(g, sorted_nodes)


print("Sigmoid Network Output = {}".format(sigmoid_ouput))
