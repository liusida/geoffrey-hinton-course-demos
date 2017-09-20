"""
This demo is based on Lecture 1
"A very simple way to recognize handwritten shapes"

"""

import time
import numpy as np
import matplotlib.pyplot as plt

#context can help the program locate the dataset directory, and also load some useful tools
import context

img_objects = []
def show(image, label, weights, prediction, ax):
    """update the image show us the progress"""
    global img_objects
    if len(img_objects)==0:
        for i in range(10):
            _img = ax[0, i].imshow(weights[i].reshape(28,28), cmap='gray')
            img_objects.append(_img)
        _img = ax[1, 5].imshow(image.reshape(28,28), cmap='gray')
        img_objects.append(_img)
    else:
        for i in range(10):
            img_objects[i].set_data(weights[i].reshape(28,28))
            img_objects[i].set_clim(vmin=0, vmax=np.max(weights[i]))
            img_objects[10].set_data(image.reshape(28,28))
    ax[0,5].set_title('truth: %d, predict: %d'%(np.argmax(label), prediction))

def predict(image, weights):
    """
    predict what digit is in the image
    :param image: the image you want to predict. must be 28x28 size.
    :param weights: using the trained weights.
    :return: the prediction, one digit 0 - 9
    """
    result = weights * image
    result = np.sum(result, axis=1)
    return np.argmax(result)

def train(image, labels, weights, learning_rate=0.01):
    """
    use image and label to adjust the weights
    "Show the network an image and increment the weights from active pixels to the correct class."
    "Then decrement the weights from active pixels to whatever class the network guesses."
    :param image: the image for training
    :param labels: the label of that image. one-hot coded.
    :param weights: the trained weights, and also this weights will be adjust.
    :param learning_rate: any learning rate will be ok, because the scale will be determined by this rate.
    :return: the prediction and weights
    """
    label = np.argmax(labels)
    y_hat = predict(image, weights)
    active_img = image > np.mean(image)
    weights[label, active_img ] += learning_rate
    if y_hat != label:
        weights[y_hat, active_img ] -= learning_rate
    return y_hat, weights

def main():
    # get MNIST dataset ( using tensorflow ancillary tools )
    # if you don't have tensorflow, you can download MNIST dataset at
    # http://yann.lecun.com/exdb/mnist/
    # and place them in the right directory: ./data/MNIST/
    # and deal with it by hand. ( I really don't like to do that. )
    dataset = context.mnist.get_mnist()

    # initialize weights as 0s
    weights = np.zeros([10, 784])

    # preparing canvas
    plt.ion()
    fig, ax = plt.subplots(ncols=10, nrows=2)
    for a in ax:
        for b in a:
            b.axis('off')
    plt.show()

    # main training process
    for epoch in range(500):
        # tensorflow gives this nice next_batch implementation.
        images, labels = dataset.train.next_batch(10)
        for i,_ in enumerate(images):
            y_hat, weights = train(images[i], labels[i], weights)
        # show iterative image every epoch
        show(images[i], labels[i], weights=weights, prediction=y_hat, ax=ax)
        # refresh the canvas and let us saw the going.
        fig.canvas.flush_events()
        time.sleep(0.5)

    # done
    ax[0,0].set_title("done")
    # leave the window open.
    plt.ioff()
    plt.show()

if __name__=='__main__':
    main()
