"""
This demo is based on Lecture 3
"A simple example of overfitting"

"""
import copy
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ===========================================
# prepare data
def prepare_data(num_total=50, true_weight=10, true_bias=50):
    true_weight = np.random.randn(1) * 50
    true_bias = np.random.randn(1) * 50
    X = np.random.randn(num_total)
    Y = X * true_weight + true_bias + np.random.randn(num_total) * 30
    return X,Y


class simple:
    # ===========================================
    # forward propagation
    def predict(self, X):
        Y_hat = X * self.weights + self.bias
        return Y_hat

    def forward_propagation(self, X, Y):
        Y_hat = self.predict(X)
        loss = np.mean(np.square(Y-Y_hat))/2
        return loss, Y_hat

    # ===========================================
    # backward propagation
    def backward_propagation(self, X, Y, Y_hat):
        delta_Y = Y - Y_hat
        delta_weights = np.mean(delta_Y * X)
        delta_bias = np.mean(delta_Y)
        return delta_weights, delta_bias

    # ===========================================
    # fit
    def fit(self, X, Y, iteration=1000, learning_rate=0.01):
        self.weights = 0
        self.bias = 0
        for i in range(iteration):
            loss, Y_hat = self.forward_propagation(X, Y)
            delta_weights, delta_bias = self.backward_propagation(X, Y, Y_hat)
            self.weights += learning_rate * delta_weights
            self.bias += learning_rate * delta_bias

class complicated:
    """
    This Deep Neural Network Architecture:
    input -> logistic layer -> linear layer -> output
    """
    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))

    def predict(self, X, cache=1):
        Z1 = np.matmul(X, self.weights['1']) + self.bias['1']
        A1 = self.sigmoid(Z1)
        Z2 = np.matmul(A1, self.weights['2']) + self.bias['2']
        A2 = Z2
        Z3 = np.matmul(A2, self.weights['3']) + self.bias['3']
        Y_hat = Z3
        if cache:
            self.Z1 = Z1 #[m, 10]
            self.A1 = A1 #[m, 10]
            self.Z2 = Z2 #[m, 7]
            self.A2 = A2 #[m, 7]
            self.Z3 = Z3 #[m, 1]
        return Y_hat

    def forward_propagation(self, X, Y):
        Y_hat = self.predict(X)
        loss = np.mean(np.square(Y-Y_hat))/2
        return loss, Y_hat

    def backward_propagation(self, X, Y, Y_hat):
        m = X.shape[0]
        delta_weights = self.weights.copy()
        delta_bias = self.bias.copy()

        delta_E_Z3 = Y - self.Z3
        delta_weights['3'] = 1/m * np.matmul( np.transpose(self.A2) , delta_E_Z3 )
        delta_bias['3'] = np.mean(delta_E_Z3, axis=0)

        delta_E_A2 = np.matmul( delta_E_Z3, np.transpose(self.weights['3']) )
        delta_weights['2'] =1/m * np.matmul( np.transpose(self.A1), delta_E_A2 )
        delta_bias['2'] = np.mean(delta_E_A2, axis=0)

        delta_E_A1 = np.matmul( delta_E_A2, np.transpose(self.weights['2']) )
        delta_E_Z1 = delta_E_A1 * self.A1 * ( 1 - self.A1)
        delta_weights['1'] = 1/m * np.matmul( np.transpose(X), delta_E_Z1 )
        delta_bias['1'] = np.mean(delta_E_Z1, axis=0)

        for ind in ['1','2','3']:
            assert(delta_weights[ind].shape == self.weights[ind].shape)
            assert(delta_bias[ind].shape == self.bias[ind].shape)

        return delta_weights, delta_bias

    def fit(self, X, Y, num_logistic_layer=30, num_linear_layer=30, iteration=100, learning_rate=0.001):
        self.weights = {
            '1': np.random.randn(1, num_logistic_layer),
            '2': np.random.randn(num_logistic_layer, num_linear_layer),
            '3': np.random.randn(num_linear_layer, 1)
        }
        self.bias = {
            '1': np.zeros(num_logistic_layer),
            '2': np.zeros(num_linear_layer),
            '3': np.zeros(1)
        }
        for i in range(iteration):
            loss, Y_hat = self.forward_propagation(X, Y)
            if i*10%iteration==0:
                print("%d/10 loss: %f"%(i*10//iteration, loss))
            delta_weights, delta_bias = self.backward_propagation(X, Y, Y_hat)
            for ind in ['1', '2', '3']:
                self.weights[ind] += learning_rate * delta_weights[ind]
                self.bias[ind] += learning_rate * delta_bias[ind]
        pass

# ===========================================
# Show
class show:
    def __init__(self):
        self.saved_line = {}

    def show_target(self, ax, X, Y):
        ax.scatter(X, Y, color='steelblue')

    def show_result(self, ax, X, Y, color='green'):
        X = np.squeeze(X)
        Y = np.squeeze(Y)
        ind_sorted = np.argsort(X)
        if color in self.saved_line:
            self.saved_line[color].set_data(X[ind_sorted], Y[ind_sorted])
        else:
            self.saved_line[color], = ax.plot(X[ind_sorted], Y[ind_sorted], color=color)

# ===========================================
# main process
def main():
    plt.ion()
    plt.show()
    fig = plt.figure(figsize=[10,5])
    ax = fig.gca()
    ax.set_title("Overfitting")

    X,Y = prepare_data(num_total=5)
    my_show = show()
    my_show.show_target(ax, X, Y)

    s = simple()
    s.fit(X, Y)
    my_show.show_result(ax, X, s.predict(X), color='green')

    # complicated model need lots of matmul, so choose a better form. X.shape [m,1]. same as tensorflow.
    X_linspace = np.linspace(np.min(X), np.max(X), num=100)
    X = X.reshape(-1, 1)
    X_linspace = X_linspace.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    c = complicated()
    for i in range(10):
        c.fit(X,Y, iteration=10000)
        my_show.show_result(ax, X_linspace, c.predict(X_linspace), color='red')
        fig.canvas.draw()


    plt.ioff()
    plt.show()

if __name__=='__main__':
    main()