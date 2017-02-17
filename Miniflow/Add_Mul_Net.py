from Miniflow import *


# Define inputs
x, y, z = Input(), Input(), Input()

f = Add(x,y,z)

d = Mul(x,y)

feed_dict = {x: 10,y: 5,z: 3}

sorted_nodes = topological_sort(feed_dict)
add_output = forward_pass(f, sorted_nodes)
mul_output = forward_pass(d, sorted_nodes)

print("Sum of {} + {} + {} = {} ".format(feed_dict[x],feed_dict[y],feed_dict[z], add_output))
print("Mul output {}".format(mul_output))


