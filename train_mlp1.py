import mlp1 as ml
import utils as ut
import random
import numpy as np

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}

EPOCHS = 40
ETA = 0.05
HIDDEN_SIZE = 20

def feats_to_vec(features):
    # YOUR CODE HERE.
    # # Should return a numpy vector of features.
    feats_vec = np.zeros(len(ut.F2I))
    for bigram in features:
        if bigram in ut.F2I:
            feats_vec[ut.F2I[bigram]] += 1
    #normalization
    num_of_matches = np.sum(feats_vec)
    return np.divide(feats_vec,num_of_matches)


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        x = feats_to_vec(features)  # convert features to a vector.
        y = ut.L2I[label]  # convert the label to number if needed.
        y_hat = ml.predict(x,params)
        if (y==y_hat):
            good +=1
        else:
            bad +=1

    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in xrange(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = ut.L2I[label]  # convert the label to number if needed.
            loss, grads = ml.loss_and_gradients(x, y, params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            params[0] -=learning_rate*grads[0]
            params[1] -=learning_rate*grads[1]
            params[2] -=learning_rate*grads[2]
            params[3] -=learning_rate*grads[3]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params

def run_test(test_data, params):
    pred_file = open("test.pred", 'w')
    for label, features in test_data:
        x = feats_to_vec(features)  # convert features to a vector.
        y_hat = ml.predict(x, params)
        for key, val in ut.L2I.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
            if val == y_hat:
                label = key
                break
        pred_file.write(str(label) + "\n")
    pred_file.close()


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.

    params = ml.create_classifier(len(ut.F2I), HIDDEN_SIZE,len(ut.L2I))
    trained_params = train_classifier(ut.TRAIN, ut.DEV, EPOCHS, ETA, params)
    run_test(ut.TRAIN,trained_params)
