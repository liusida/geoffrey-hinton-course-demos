"""
This demo is based on Lecture 3
"The error surface for a linear neuron"

"For a linear neuron with a squared error, it is a quadratic bowl. "
"""
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Reuse previous demo, so we don't need to implement training/predicting function again.
demo = __import__('demo-lecture-3-1')


def calculate_loss(X, Y, weights):
    Y_hat = np.matmul( np.transpose(X), weights )
    loss = np.mean(np.square(Y - Y_hat)) / 2
    return loss


# ====================================
# Show
# This file is mainly about showing the error quadratic bowl.
# In order to draw the map, we need the true_weights, otherwise we cannot know where is the map center.
# In slides it says, "Vertical cross-sections are parabolas."
def show_quadratic_parabolas(ax, ind_weight, X, Y, true_weights, plot_steps, plot_scale):
    ax.cla()
    ax.set_title('loss w.r.t. weight %d'%ind_weight)
    loss_list = []
    weights = np.copy(true_weights)
    w = weights[ind_weight]
    margin_left = plot_steps//2 * plot_scale
    margin_right = (plot_steps - plot_steps//2) * plot_scale - 1
    plot_x = np.linspace(w-margin_left, w+margin_right, num=plot_steps)

    for i in range(plot_steps):
        weights[ind_weight] = w - margin_left + i*plot_scale
        loss = calculate_loss(X, Y, weights)
        loss_list.append(loss)

    ax.plot(plot_x, loss_list)

def show_current_status_2d(ax, ind_weight, X, Y, weights, color):
    loss = calculate_loss(X, Y, weights)
    # Because other two dimension of weights are changing, so the dots are not on the curve! lol!
    ax.scatter(weights[ind_weight], loss, s=0.2, c=color)


# In slides it says, "Horizontal cross-sections are ellipses."
def show_contour(ax, ind_weight_horizontal, ind_weight_vertical, X, Y, true_weights, plot_steps, plot_scale=1):
    ax.cla()
    ax.set_title('loss contour: only show weights: %d and %d'%(ind_weight_horizontal, ind_weight_vertical))
    weights = np.copy(true_weights)
    w_h = weights[ind_weight_horizontal]
    w_v = weights[ind_weight_vertical]
    margin_left = plot_steps//2 * plot_scale
    margin_right = (plot_steps - plot_steps//2) * plot_scale - 1

    loss_list = np.zeros([plot_steps, plot_steps])
    contour_x = np.linspace(w_h-margin_left, w_h+margin_right, num=plot_steps)
    contour_y = np.linspace(w_v-margin_left, w_v+margin_right, num=plot_steps)

    for i in range(plot_steps):
        weights[ind_weight_horizontal] = w_h - margin_left + i*plot_scale
        for j in range(plot_steps):
            weights[ind_weight_vertical] = w_v-margin_left+j*plot_scale
            loss = calculate_loss(X, Y, weights)
            loss_list[j,i] = loss

    contour_x, contour_y = np.meshgrid(contour_x, contour_y)
    ax.contour(contour_x, contour_y, loss_list, 50)


def show_current_status_3d(ax, ind_weight_horizontal, ind_weight_vertical, weights, last_weights, color):
    if last_weights is not None:
        ax.scatter( last_weights[ind_weight_horizontal], last_weights[ind_weight_vertical] , c='#FFFFFF', alpha=0.4)
        ax.arrow( x=last_weights[ind_weight_horizontal], y=last_weights[ind_weight_vertical],
                  dx=weights[ind_weight_horizontal]-last_weights[ind_weight_horizontal],
                  dy=weights[ind_weight_vertical]-last_weights[ind_weight_vertical],
                  alpha = 0.2, color=color
                  )
    ax.scatter( weights[ind_weight_horizontal], weights[ind_weight_vertical] , c=color)
    return weights


def main():
    plt.ion()
    plt.show()
    fig = plt.figure(figsize=[10, 5])
    gs = matplotlib.gridspec.GridSpec(3, 4)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])
    ax_contour = fig.add_subplot(gs[:,1:])

    true_weights = [150, 50, 100]
    plot_steps = 20
    plot_scale = 100

    # total updating steps will be showed.
    total_step = 20

    # use rescale=10, I set the first feature to a large scale.
    # I have observed that, if I still use the sameple learning rate, the loss value will
    #   explode along the direction of first feature!
    # if I turn down the learning rate, it finally can converge, but very slow very slow with respect to the second
    #   and third features.
    X,Y = demo.prepare_data(total_num=20, true_weights=true_weights, rescale=1)
    print(X,Y)

    show_quadratic_parabolas(ax0, 0, X, Y, true_weights, plot_steps, plot_scale)
    show_quadratic_parabolas(ax1, 1, X, Y, true_weights, plot_steps, plot_scale)
    show_quadratic_parabolas(ax2, 2, X, Y, true_weights, plot_steps, plot_scale)
    show_contour(ax_contour, 0, 1, X, Y, true_weights, plot_steps, plot_scale)
    plt.tight_layout()

    # Online versus batch learning
    # 1. Online learning
    last_weights = None
    weights = [250,1000,0]
    epoch = total_step // X.shape[1]
    for i in range(epoch):
        for j in range(X.shape[1]):
            # here we do not do vectorized training, so we can see the oscillation
            x = X[:,j].reshape(-1,1)
            y = Y[j].reshape(-1,1)
            last_weights = show_current_status_3d(ax_contour, 0, 1, weights, last_weights, 'steelblue')
            show_current_status_2d(ax0, 0, X, Y, weights, 'blue')
            show_current_status_2d(ax1, 1, X, Y, weights, 'blue')
            show_current_status_2d(ax2, 2, X, Y, weights, 'blue')
            weights, Y_hat = demo.train(x, y, weights, learning_rate=1e-2)

            fig.canvas.draw()
            time.sleep(0.1)

    # 2. Batch learning
    last_weights = None
    weights = [250,1000,0]
    epoch = total_step
    for i in range(epoch):
        last_weights = show_current_status_3d(ax_contour, 0, 1, weights, last_weights, 'orange')
        show_current_status_2d(ax0, 0, X, Y, weights, 'orange')
        show_current_status_2d(ax1, 1, X, Y, weights, 'orange')
        show_current_status_2d(ax2, 2, X, Y, weights, 'orange')
        weights, Y_hat = demo.train(X, Y, weights, learning_rate=1e-2)

        fig.canvas.draw()
        time.sleep(0.1)


    plt.ioff()
    plt.show()

if __name__=='__main__':
    main()