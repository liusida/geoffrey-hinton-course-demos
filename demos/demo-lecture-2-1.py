"""
This demo is based on Lecture 2
"Perceptrons: The first generation of neural networks"

"""
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# preparing data:
# let's say we have some positive data: y = a * (x-b)^2 + c,
# and we generate some negative data.
# see if the perceptrons can learn the decision boundary.
def prepare_data(total_num=100, a=0, b=0, c=0):
    pos_num = total_num//2
    neg_num = total_num - pos_num

    # generate positive samples:
    if a==0 and b==0 and c==0:
        a,b,c = np.random.randn(3)
    X_pos = np.random.randn(pos_num)
    Y_pos = a * np.square(X_pos-b) + c
    labels_pos = np.ones(pos_num)

    # generate negative samples:
    # Notice: we can not generate negative point on the other side of positive curve. Preceptrons can not learn that.
    X_neg = np.random.randn(neg_num)
    offset_neg = np.random.rand(neg_num) + 5
    Y_neg = a * np.square(X_neg-b) + c + offset_neg
    labels_neg = np.zeros(neg_num)

    # concat and shuffle features and labels.
    X = np.concatenate((X_pos, X_neg))
    Y = np.concatenate((Y_pos, Y_neg))
    labels = np.concatenate((labels_pos, labels_neg))

    random_index = np.random.permutation(X.shape[0])
    X = X[random_index]
    Y = Y[random_index]
    labels = labels[random_index]

    # 3 features(x, x^2, y), with 1 at the end to behave with the last weight as bias
    X = X.reshape(1,-1)
    Y = Y.reshape(1,-1)
    features = np.concatenate((X, np.square(X), Y, np.ones(X.shape)), axis=0)

    return features,labels

# Predict
# y_hat = transpose(X) * W + b
# if y_hat > threshold, then shoot.
def predict(features, weights):
    weights = weights.reshape(-1,1)
    Y_hat = np.matmul( np.transpose(features), weights )
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
    y_hat = predict(feature, weights)
    if y_hat != label:
        if y_hat==1:
            weights -= feature
        else:
            weights += feature
    return weights

# ====================================
#  Show functions
saved_curve = None
def show_curve(ax, X, Y_hat, color='red'):
    """show the learned decision boundary in 2d canvas"""
    global saved_curve
    sorted_index = X.argsort()
    if saved_curve == None:
        saved_curve, = ax.plot(X[sorted_index], Y_hat[sorted_index], color=color)
    else:
        saved_curve.set_data(X[sorted_index], Y_hat[sorted_index])

def show_target_neg_scatter(ax, X, Y):
    ax.scatter(X, Y, color='green')

def show_target_pos_scatter(ax, X, Y):
    ax.scatter(X, Y, color='steelblue')

saved_active_scatter = None
def show_active_scatter(ax, x, y):
    """show which point is training now."""
    global saved_active_scatter
    if saved_active_scatter:
        saved_active_scatter.remove()
    saved_active_scatter = ax.scatter(x, y, color='yellow')

def show_3d_scatter(ax, features, labels):
    ax.scatter(features[0,labels==1], features[1,labels==1], features[2,labels==1], color='steelblue')
    ax.scatter(features[0,labels==0], features[1,labels==0], features[2,labels==0], color='green')

saved_surface = None
def show_3d_decision_boundary(ax, X, Y, Z):
    global saved_surface
    max_points = 10
    index_max = np.argmax(X)
    index_min = np.argmin(X)
    X = np.concatenate(([X[index_min]], X[:max_points], [X[index_max]]))
    Y = np.concatenate(([Y[index_min]], Y[:max_points], [Y[index_max]]))
    Z = np.concatenate(([Z[index_min]], Z[:max_points], [Z[index_max]]))
    sorted_index = X.argsort()
    X = X[sorted_index]
    Y = Y[sorted_index]
    Z = Z[sorted_index]
    X, Y = np.meshgrid(X, Y)
    if saved_surface:
        saved_surface.remove()
    saved_surface = ax.plot_surface(X, Y, Z, alpha=0.6, color='red')

# ====================================
# Main process
def main():
    plt.ion()
    plt.show()
    fig = plt.figure(figsize=[10,5])

    ax_2d = fig.add_subplot(121)
    ax_3d = fig.add_subplot(122, projection='3d')

    total_num = 200
    train_num = 200

    features,labels = prepare_data(total_num=total_num)

    features_train = features[:,:train_num]
    labels_train = labels[:train_num]

    #initialize weights: 3 feature weights and 1 bias
    weights = np.random.randn(4)

    # show target curve
    # in 2d canvas, i, j representation:
    # in 3d canvas, i, j, k representation:
    axis_i = 0
    axis_j = 2
    axis_hidden = 1
    # Unfortunately, axis_j cannot represent data x,
    # because there will be two kind of value of x respect to a linear y, so the decision boundary will be broken.

    show_target_pos_scatter(ax_2d, features[axis_i,labels==1], features[axis_j,labels==1])
    show_target_neg_scatter(ax_2d, features[axis_i,labels==0], features[axis_j,labels==0])
    show_3d_scatter(ax_3d, features, labels)
    for i in range(100):
        for i_, feature in enumerate(np.transpose(features_train)):
            label = labels_train[i_]
            weights = train(feature=feature, label=label, weights=weights)
            show_active_scatter(ax_2d, feature[axis_i], feature[axis_j] )

            if i_%(train_num//10)==0:
                # every iteration show 10 steps
                X = features[axis_i]
                X_hidden = features[axis_hidden]
                Y_hat = -(X * weights[axis_i] + X_hidden * weights[axis_hidden] + weights[3]) / weights[axis_j]
                show_curve(ax_2d, X, Y_hat)

                show_3d_decision_boundary(ax_3d, X, X_hidden, Y_hat )
                ax_2d.set_title('weights: %f,%f,%f,%f'%(weights[0], weights[1], weights[2], weights[3]))
                fig.canvas.draw()
                time.sleep(0.1)

        Y_hat = predict(features, weights)
        if (np.mean(Y_hat==labels)==1):
            break

    print('donw in ',i,'epoches')
    plt.ioff()
    plt.show()

if __name__=='__main__':
    main()