"""
This demo is based on Lecture 2
"Perceptrons: The first generation of neural networks"

"""
import time
import numpy as np
import matplotlib.pyplot as plt

# context can help the program locate the dataset directory, and also load some useful tools
# import context

# preparing data:
# let's say we have a decision boundary y = a * (x-b)^2 + c, see if the perceptrons can learn this curve
# a, b, c can be random numbers. so that we saw a quadratic curve, we can give the formula.
# generate positive samples:
# generate negative samples:
# concat and shuffle features and labels.

# Decide what are features?
# required: x, x^2
# optional: x^3, sqrt(x)

# Predict
# y_hat = transpose(X) * W + b
# if y_hat > threshold, then shoot.

# Train
# if predict is right, do nothing.
# if predict is wrong:
#   if predict is 1:
#       W += X
#   else:
#       W -= X

# Show
# plot the target curve, using the secret a,b,c we have.
# plot a moving curve, using trained weights, to see if it will converge to the target curve.
