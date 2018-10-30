import numpy as np
import loglinear as ll

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}
PRECISION = 1e-4
h_s = []
z_s = []

def classifier_output(x, params):
    # YOUR CODE HERE.
    h = x
    h_s.append(h)
    for i in range(0,len(params),2):
        z = np.dot(h,params[i]) +params[i+1]
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
    gradients.append(np.outer(grad_so_far, h_s.pop()))

    #compute grad of all params
    W_s = [num for num in params if params.index(num) % 2 == 1]
    b_s = [num for num in params if params.index(num) % 2 == 0]
    W_s = W_s.reverse()
    b_s = b_s.reverse()
    index = 0
    for W,b in zip(params[0:-2:2], params[1:-1:2]):
        z_i = z_s.pop()
        w_i_plus_one =W_s[index]
        h_i_minus_one = h_s.pop()

        dz_dh = w_i_plus_one
        dh_dz = 1-np.square(np.tanh(z_i))
        grad_so_far = np.dot(np.dot(grad_so_far,dz_dh),dh_dz)

        dz_dw = h_i_minus_one
        #grad of w
        gradients.append(np.dot(grad_so_far, dz_dw))
        #grad of b
        gradients.append(grad_so_far)
        index +=1


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
    print dims
    params = []
    for dim1,dim2 in zip(dims,dims[1:]):
        params.append(np.random.uniform(-PRECISION, PRECISION,[dim1,dim2]))
        params.append(np.random.uniform(-PRECISION,PRECISION,dim2))
    return params

