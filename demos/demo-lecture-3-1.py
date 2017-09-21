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
import numpy as np
import matplotlib.pyplot as plt

# ====================================
# set true weights
# prepare data
def prepare_data(total_num=10, true_weights=[150, 50, 100]):
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
    delta_weights = learning_rate * X * delta_Y
    # delta_weights = np.mean(delta_weights, axis=1) [2]
    delta_weights = np.mean(delta_weights, axis=1)
    weights = weights + delta_weights
    return weights

# ==
# Show
show_error_curve_init = None
def show_error_curve():
    global show_error_curve_init, loss_list
    if show_error_curve_init:
        #show_error_curve_init.remove()
        pass
    show_error_curve_init = plt.plot(range(len(loss_list)), loss_list)

def show_weights(weights):
    pass

# ====================================
# main
def main():
    plt.ion()
    plt.show()
    true_weights = [150, 50, 100]
    X,Y = prepare_data(total_num=50, true_weights=true_weights)
    print("Training dataset: \nX[0]:", X[:, 0])
    print("Y[0]:", Y[0])
    # initialize guessing weights
    weights = [50, 50, 50]
    print("initial Y_hat[0]:", predict(X, weights)[0])

    epoch = 10
    for i in range(epoch):
        weights = train(X, Y, weights, learning_rate=1e-2)
        print("during training Y_hat[0]:", predict(X, weights)[0])

    show_error_curve()
    print("Learned weights:", weights)
    print("True weights:", true_weights)


    plt.ioff()
    plt.show()

if __name__=='__main__':
    main()