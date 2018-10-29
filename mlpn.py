import numpy as np
import loglinear as ll

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}
PRECISION = 1e-4
h_s = []
z_s = []

def classifier_output(x, params):
    # YOUR CODE HERE.
    params_copy = params.copy()
    h = x
    for i in range(0,len(params),2):
        z = np.dot(params[i],h) +params[i+1]
        z_s.append(z)
        h = np.tanh(z)
        h_s.append(h)
    h_s.pop()
    probs = ll.softmax(z)
    z_s.pop()
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE
    probs = classifier_output(x, params)  # pred vec
    loss = -np.log(probs[y])
    gradients = []
    y_one_hot = np.zeros(len(probs))
    y_one_hot[y] = 1
    grad_so_far = -(y_one_hot - probs)

    #grad of bn
    gradients.append(grad_so_far)
    # grad of wn
    gradients.append(np.dot(grad_so_far, h_s.pop()))
    #grad of all params

    for W,b in zip(params, params[1:]):
        z_i =
        w_i_plus_one =
        h_i_minus_one =

        dh_dz = 1-np.square(np.tanh(z_i))
        dz_dh = w_i_plus_one
        grad_so_far = np.dot(np.dot(grad_so_far,dh_dz),dz_dh)


        dz_dw = h_i_minus_one




    for i in range(len(params)-1, -1, -1):
    gradients.append(np.dot(grad_so_far,hidden_layers[]))
    return loss, gradients.reverse()

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for i in range(0,len(dims),2):
        params[i] = np.random.uniform(-PRECISION, PRECISION,[dims[i],dims[i+1]])
        params[i+1] = np.random.uniform(-PRECISION,PRECISION,dims[i+1])
    return params

