import numpy as np
import loglinear as ll

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    h = x
    for i in range(0,len(params),2):
        z = np.dot(h,params[i]) +params[i+1]
        h = np.tanh(z)
    probs = ll.softmax(z)
    return probs

def fp(x, params):
    # make the layers params
    h_s = []
    z_s = []
    h = x
    h_s.append(h)
    for i in range(0, len(params), 2):
        z = np.dot(h, params[i]) + params[i + 1]
        z_s.append(z)
        h = np.tanh(z)
        h_s.append(h)
    h_s.pop()
    z_s.pop()
    return h_s, z_s


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
    h_s, z_s = fp(x,params)
    loss = -np.log(probs[y])
    gradients = []
    y_one_hot = np.zeros(len(probs))
    y_one_hot[y] = 1
    grad_so_far = -(y_one_hot - probs)

    # grad of wn
    gradients.append(np.outer(h_s.pop(),grad_so_far))

    #grad of bn
    gradients.append(np.copy(grad_so_far))
    #compute grad of all params
    for i, (w, b) in enumerate(zip(params[-2::-2], params[-1::-2])):
        if (len(z_s)!=0):
            z_i = z_s.pop()
            w_i_plus_one =w
            if (len(h_s)!=0):
                h_i_minus_one = h_s.pop()
                #calcelate gradients
                dz_dh = w_i_plus_one
                dh_dz = 1-np.square(np.tanh(z_i))
                dz_dw = h_i_minus_one
                grad_so_far = np.dot(grad_so_far,np.transpose(dz_dh)) * dh_dz

                #grad of w
                gradients.append(np.outer(dz_dw,grad_so_far))
                #grad of b
                gradients.append(np.copy(grad_so_far))
    rev_grad = []
    for w, b in zip(gradients[0::2], gradients[1::2]):
        rev_grad.append(b)
        rev_grad.append(w)

    return loss, list(reversed(rev_grad))

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
        eps = np.sqrt(6) / (np.sqrt(dim1 + dim2))
        params.append(np.random.uniform(-eps, eps,[dim1,dim2]))
        eps = np.sqrt(6) / (np.sqrt(dim2))
        params.append(np.random.uniform(-eps,eps,dim2))
    return params

