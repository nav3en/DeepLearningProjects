from Miniflow import *


# Define inputs
inputs, weights, bias = Input(), Input(), Input()

f1 = Linear(inputs, weights, bias)


feed_dict = {inputs: [1,2,3],
             weights: [2,2,2],
             bias: 2}

sorted_nodes = topological_sort(feed_dict)
linear_ouput= forward_pass(f1, sorted_nodes)


print("Simple Linear Network Output = {}".format(linear_ouput))

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])


X, W, b = Input(), Input(), Input()

feed_dict = {X: X_, W: W_, b: b_}

f2 = Linear(X,W,b)
sorted_nodes = topological_sort(feed_dict)
linear_ouput= forward_pass(f2, sorted_nodes)


print("Slightly complex Linear Network Output = {}".format(linear_ouput))
