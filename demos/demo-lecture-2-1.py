"""
This demo is based on Lecture 2
"Perceptrons: The first generation of neural networks"

"""
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# context can help the program locate the dataset directory, and also load some useful tools
# import context

# preparing data:
# let's say we have a decision boundary y = a * (x-b)^2 + c, see if the perceptrons can learn this curve
# a, b, c can be random numbers. so that we saw a quadratic curve, we can give the formula.
# generate positive samples:
# generate negative samples:
# concat and shuffle features and labels.
def prepare_data(total_num=100, a=0, b=0, c=0):
    if a==0 and b==0 and c==0:
        a,b,c = np.random.randn(3)
    X_pos = np.random.randn(total_num)
    Y_pos = a * np.square(X_pos-b) + c
    labels_pos = np.ones(total_num)

    X_neg = np.random.randn(total_num)
    offset_neg = np.random.rand(total_num) + 5
    #offset_neg[offset_neg>0] += 1
    #offset_neg[offset_neg<=0] -= 1
    Y_neg = a * np.square(X_neg-b) + c + offset_neg
    labels_neg = np.zeros(total_num)

    X = np.concatenate((X_pos, X_neg))
    Y = np.concatenate((Y_pos, Y_neg))
    labels = np.concatenate((labels_pos, labels_neg))

    random_index = np.random.permutation(X.shape[0])
    X = X[random_index]
    Y = Y[random_index]
    labels = labels[random_index]

    X = X.reshape(1,-1)
    Y = Y.reshape(1,-1)
    features = np.concatenate((X, np.square(X), Y, np.ones(X.shape)), axis=0)

    return features,labels

# Decide what are features?
# required: x, x^2
# optional: x^3, sqrt(x)

# Predict
# y_hat = transpose(X) * W + b
# if y_hat > threshold, then shoot.
def predict(features, weights):
    weights = weights.reshape(-1,1)
    Y_hat = np.matmul( np.transpose(features), weights )
    #print("calculate result: ", Y_hat)
    Y_hat[Y_hat>0] = 1
    Y_hat[Y_hat<=0] = 0
    Y_hat = np.transpose(Y_hat).reshape(-1)
    return Y_hat

# Train
# if predict is right, do nothing.
# if predict is wrong:
#   if predict is 1:
#       W += X
#   else:
#       W -= X
def train(feature, label, weights):
    #print("==== training detail ===== ")
    #print("label: ", label)
    #print("feature: ", feature)
    #print("weights: ", weights)

    y_hat = predict(feature, weights)
    #print("prediction: ", y_hat)
    if y_hat != label:
        if y_hat==1:
            weights -= feature
        else:
            weights += feature
    #print("weights after update: ", weights)

    return weights

# Show
# plot the target curve, using the secret a,b,c we have.
# plot a moving curve, using trained weights, to see if it will converge to the target curve.
saved_curve = None
def show_curve(ax, X, Y, weights, color='steelblue'):
    global saved_curve
    sorted_index = X.argsort()
    Y_hat = -(X*weights[0]+np.square(X)*weights[1]+weights[3]) / weights[2]
    if saved_curve == None:
        saved_curve, = ax.plot(X[sorted_index], Y_hat[sorted_index], color=color)
    else:
        saved_curve.set_data(X[sorted_index], Y_hat[sorted_index])

def show_target_curve(ax, X,Y):
    sorted_index = X.argsort()
    ax.plot(X[sorted_index], Y[sorted_index], color='steelblue')

def show_target_neg_scatter(ax, X, Y):
    ax.scatter(X, Y, color='green')

def show_3d_scatter(ax, features, labels):
    ax.scatter(features[0,labels==1], features[1,labels==1], features[2,labels==1], color='steelblue')
    ax.scatter(features[0,labels==0], features[1,labels==0], features[2,labels==0], color='green')

saved_surface = None
def show_3d_decision_boundary(ax, weights, XY_range):
    """TODO: here must be still have some errors in it. the decision surface is not right."""
    global saved_surface
    X = np.transpose(XY_range)[0]
    Y = np.transpose(XY_range)[1]
    X, Y = np.meshgrid(X, Y)
    X_2 = np.square(X)
    Z = -(X*weights[0] + Y*weights[2] + weights[3]) / weights[1]
    if saved_surface:
        saved_surface.remove()
    saved_surface = ax.plot_surface(X, Y, Z, alpha=0.6, color='red')

# Main process
def main():
    plt.ion()
    plt.show()
    fig = plt.figure()

    ax_2d = fig.add_subplot(121)
    ax_3d = fig.add_subplot(122, projection='3d')

    total_num = 300
    train_num = 300

    features,labels = prepare_data(total_num=total_num)
    X_min = np.min(features[0])
    X_max = np.max(features[0])
    Y_min = np.min(features[2])
    Y_max = np.max(features[2])
    XY_range = np.array([[X_min, Y_min], [X_max, Y_min], [X_min, Y_max], [X_max, Y_max]])

    features_train = features[:,:train_num]
    labels_train = labels[:train_num]

    #initialize weights: 3 feature weights and 1 bias
    weights = np.zeros(4)

    # show target curve
    show_target_curve(ax_2d, features[0,labels==1], features[2,labels==1])
    show_target_neg_scatter(ax_2d, features[0,labels==0], features[2,labels==0])
    show_3d_scatter(ax_3d, features, labels)
    for i in range(100):
        for i_, feature in enumerate(np.transpose(features_train)):
            label = labels_train[i_]
            weights = train(feature=feature, label=label, weights=weights)

        # every iteration show predict curve
        Y_hat = predict(features, weights)
        show_curve(ax_2d,features[0,Y_hat==1], features[2,Y_hat==1], weights, color='red')
        #show_3d_decision_boundary(ax_3d, weights, XY_range )
        #fig.canvas.draw()
        #time.sleep(0.5)

        Y_hat = predict(features, weights)
        if (np.mean(Y_hat==labels)==1):
            break

    print(i,'epoches')
    plt.ioff()
    plt.show()

if __name__=='__main__':
    main()