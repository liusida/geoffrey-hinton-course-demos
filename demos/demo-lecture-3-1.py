"""
This demo is based on Lecture 3
"A toy example to illustrate the iterative method"

Story:
Each day you get lunch at the cafeteria.
Your diet consists of fish, chips, and ketchup.
You get several portions of each.
The cashier only tells you the total price of the meal.
After several days, you should be able to figure out the price of each portion.

"""
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# ====================================
# set true weights
# prepare data
def prepare_data(total_num, true_weights):
    X = np.random.random_integers(low=0, high=10, size=[3, total_num])
    Y = np.matmul( np.transpose(X), true_weights )
    return X,Y


# ====================================
# predict
def predict(X, weights):
    Y_hat = np.matmul( np.transpose(X), weights )
    return Y_hat

# ====================================
# train
# Experience: I think it's better to start implementing with a tiny learning_rate.
#             Because if learning rate is too large, the learning cannot converge.
#             At the very start, we don't know the learning does not converge because the
#             learning rate is too large or there are bugs in implementation.
loss_list = []
def train(X, Y, weights, learning_rate=1e-10):
    global Error_list
    Y_hat = predict(X, weights)
    # [1][2] Explanation:   in the slides, it says use sum:
    #                       "Define the error as the squared residuals summed over all."
    #                       But when the batch size is very large, it has the tendency to diverge.
    #                       So, if we don't want tune learning rate every time when batch size changed,
    #                       we'd better use mean.
    # loss = np.sum(np.square(Y - Y_hat)) / 2 [1]
    loss = np.mean(np.square(Y - Y_hat)) / 2
    loss_list.append(loss)
    delta_Y = Y - Y_hat
    print("---------------------> delta_Y[0]:", delta_Y[0])
    delta_weights = learning_rate * X * delta_Y
    # delta_weights = np.mean(delta_weights, axis=1) [2]
    delta_weights = np.mean(delta_weights, axis=1)
    print("---------------------> delta_weights:", delta_weights)
    weights = weights + delta_weights
    print("---------------------> weights after update:", weights)
    return weights, Y_hat


# ====================================
# Show
def show_error_curve(ax):
    global loss_list
    ax.cla()
    ax.set_title("loss")
    ax.plot(loss_list)

def show_weights(ax, weights, true_weights):
    ax.cla()
    ax.set_title("weights")
    ind = np.arange(len(weights))
    bar_width = 0.1
    bar_learned_weight = ax.bar(ind, weights, width=bar_width, color='orange')
    bar_target_weight = ax.bar(ind+bar_width, true_weights, width=bar_width, color='steelblue')
    ax.legend((bar_learned_weight[0], bar_target_weight[0]), ('Learned Weights', 'Target Weights'))

def show_y(ax, X, Y, Y_hat):
    ax.cla()
    ax.set_title("predictions")
    ind = np.arange(len(Y))
    bar_width = 0.4
    bar_prediction = ax.bar(ind, Y_hat, width=bar_width, color='orange')
    bar_truth = ax.bar(ind+bar_width, Y, width=bar_width, color='steelblue')
    ax.legend((bar_prediction[0], bar_truth[0]), ("Prediction", "Truth"))


# ====================================
# main
def main():
    plt.ion()
    plt.show()
    fig = plt.figure(figsize=[10, 5])
    gs = matplotlib.gridspec.GridSpec(2, 2)
    ax_loss = fig.add_subplot(gs[0,0])
    ax_weights = fig.add_subplot(gs[0,1])
    ax_y = fig.add_subplot(gs[1,:])

    true_weights = [150, 50, 100]
    X,Y = prepare_data(total_num=50, true_weights=true_weights)
    print("There are 50 examples in one batch, so I just print the first one, to see what's going on.\n")
    print("Training dataset: \nX[0]:", X[:, 0])
    print("Y[0]:", Y[0])
    print("True weights:", true_weights)

    # initialize guessing weights
    weights = [50, 50, 50]
    print("initial Y_hat[0]:", predict(X, weights)[0])

    epoch = 10
    for i in range(epoch):
        weights, Y_hat = train(X, Y, weights, learning_rate=1e-2)
        print("after training Y_hat[0]:", predict(X, weights)[0])
        print()
        show_error_curve(ax_loss)
        show_weights(ax_weights, weights, true_weights)
        show_y(ax_y, X, Y, Y_hat)
        fig.canvas.draw()
        time.sleep(0.5)

    print("Learned weights:", weights)

    plt.ioff()
    plt.show()

if __name__=='__main__':
    main()