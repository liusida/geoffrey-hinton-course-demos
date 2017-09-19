import pickle, os

GLOBALS = {
    'project_root' : ''
    }

def init(project_root):
    global GLOBALS
    GLOBALS['project_root'] = project_root

def get_mnist():
    """
    use tensorflow to get MNIST dataset
    """
    if GLOBALS['project_root']=='':
        print('please initialize project_root in GLOBALS first')
        return None
    data_path = os.path.join(GLOBALS['project_root'], 'data/MNIST/')
    pickle_path = os.path.join(data_path, 'mnist.pickle')
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            mnist = pickle.load(f)
    else:
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets(data_path, one_hot=True)
        with open(pickle_path, 'wb') as f:
            pickle.dump(mnist, f, pickle.HIGHEST_PROTOCOL)

    return mnist

def get_mini_samples():
    """
    use tensorflow to get MNIST dataset, and extract 50 mini samples.
    """
    if GLOBALS['project_root']=='':
        print('please initialize project_root in GLOBALS first')
        return None
    data_path = os.path.join(GLOBALS['project_root'], 'data/MNIST/')
    pickle_path = os.path.join(data_path, 'mnist_mini_samples.pickle')
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            mini_samples = pickle.load(f)
    else:
        mnist = get_mnist()
        mini_samples = mnist.train.next_batch(50)
        with open(pickle_path, 'wb') as f:
            pickle.dump(mini_samples, f, pickle.HIGHEST_PROTOCOL)

    return mini_samples


