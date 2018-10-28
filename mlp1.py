import numpy as np
import loglinear as ll

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

PRECISION = 1e-4

def classifier_output(x, params):
    # YOUR CODE HERE.
    W = params[0]
    b = params[1]
    U = params[2]
    b_tag = params[3]

    probs = ll.softmax(np.dot(U,(np.tanh(np.dot(W,x)+b)))+b_tag)
    return probs

def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # YOU CODE HERE
    # compute the loss
    W = params[0]
    b = params[1]
    U = params[2]
    b_tag = params[3]
    probs = classifier_output(x, params)  # pred vec
    y_one_hot = np.zeros(len(probs))
    y_one_hot[y] = 1
    gb_tag = -(y_one_hot-probs)
    gU = np.outer(gb_tag, np.tanh(np.dot(W,x)+b))
    gb = np.dot(gb_tag, U) * (1-np.square(np.tanh(np.dot(W,x)+b)))
    gW = np.outer(gb,x)
    loss = -np.log(probs[y])

    return loss, [gW, gb, gU, gb_tag]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W = np.random.uniform(-PRECISION,PRECISION,[hid_dim, in_dim])
    b = np.random.uniform(-PRECISION,PRECISION,hid_dim)
    U = np.random.uniform(-PRECISION,PRECISION,[out_dim, hid_dim])
    b_tag = np.random.uniform(-PRECISION,PRECISION,out_dim)

    params = [W,b,U,b_tag]
    return params
