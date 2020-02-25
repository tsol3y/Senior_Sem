import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import math
import datetime

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) # exp function provided by numpy can support vector operation by default
    
# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    gz = sigmoid(y)
    return gz * (1.0 - gz)

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.n1 = ni + 1 # +1 for bias node
        self.n2 = nh
        self.n3 = no

        # activations for nodes
        self.a1 = np.zeros(shape=(self.n1, 1))
        self.a2 = np.zeros(shape=(self.n2, 1))
        self.a3 = np.zeros(shape=(self.n3, 1))
        
        # same as shown in the model
        self.z2 = np.array([])
        self.z3 = np.array([])

        # create weights variables (the theta in model)
        self.w1 = self.weights_init(ni, nh)
        self.w2 = self.weights_init(nh, no)

        # to accumulate the gradient from all the train samples   
        self.Delta1 = np.zeros(shape=(self.n2, self.n1))  # for w1
        self.Delta2 = np.zeros(shape=(self.n3, self.n2+1))# for w2

    # When training neural networks, it is important to randomly initialize 
    # the parameters for symmetry breaking.
    def weights_init(self, l_in, l_out):
        eps_init = 0.12
        ret = np.random.rand(l_out, 1+l_in) * 2 * eps_init - eps_init
        return ret

    # predict output according to the given inputs
    def update(self, inputs):
        self.a1 = np.vstack(([1.0], inputs))

        # hidden activations
        self.z2 = np.dot(self.w1, self.a1)
        self.a2 = np.vstack(([1.0], sigmoid(self.z2)))

        # output activations
        self.z3 = np.dot(self.w2, self.a2)
        self.a3 = sigmoid(self.z3)
        return self.a3

    # after each iteration, we update the weights, 
    # here we use gradient descent optimization algorithm, and the learn rate is set as 1.0
    def weights_update(self, Lambda, m):
        dw1 = self.Delta1 / m
        dw2 = self.Delta2 / m
        # for all the weights, you should not apply regulization on the first column since they are for bias
        for i in range(self.w1.shape[0]):
            self.w1[i, 0] = self.w1[i, 0] - dw1[i, 0]
        for i in range(self.w1.shape[0]):
            for j in range(1, self.w1.shape[1]):
                self.w1[i, j] = self.w1[i, j] - (dw1[i, j] + (Lambda/m) * self.w1[i, j])
        
        for i in range(self.w2.shape[0]):
            self.w2[i, 0] = self.w2[i, 0] - dw2[i, 0]
        for i in range(self.w2.shape[0]):
            for j in range(1, self.w2.shape[1]):
                self.w2[i, j] = self.w2[i, j] - (dw2[i, j] + (Lambda/m) * self.w2[i, j])
                
        # this is importment, since the variable is to accumulate the gradient from each training sample
        # we should clear the accumulation after each iteration
        self.Delta1.fill(0.0)
        self.Delta2.fill(0.0)

    # distribute the cost of predict to all the nodes
    def backPropagate(self, targets):
        # calculate error terms for output
        delta3 = self.a3 - targets
        # do not forget to skip the first column which is for bias and should not be included
        delta2 = (np.dot(self.w2.T, delta3))[1:] * dsigmoid(self.z2)
        #accumulate the gradient from all the train samples 
        self.Delta1 = self.Delta1 + np.dot(delta2, self.a1.T)
        self.Delta2 = self.Delta2 + np.dot(delta3, self.a2.T)

        # calculate error, mse was used, just to show the minimization, 
        # if use other optimize method, should use cost function 
        error = 0.0
        for k in range(len(targets)):
            error += (targets[k] - self.a3[k])**2

        return np.sqrt(error) / len(targets)

    # to test a trained ANN
    def test(self, patterns):
        ac = 1
        for p in patterns:
            predict = self.update(p[0])
            print(1 + np.argmax(p[1]), 'vs.', 1 + np.argmax(predict))
            if np.argmax(p[1]) == np.argmax(predict):
                ac += 1
        print("training set accuracy: {0} %".format(100.0 * ac / len(patterns)))

    # train the model by using gradient descent optimization algorithm
    def train(self, patterns, iterations, Lambda):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error += self.backPropagate(targets)
            self.weights_update(Lambda, len(patterns))
            if i % 50 == 0:
                print('error: {0}'.format(error))

X = np.load("X.npy")
y = np.load("y.npy")
print (X.shape, y.shape)

def handwritten_visualize(X, ncol, nrow):
# handwritten figure was saved as 20*20 pixel, 
# one line in X with size 400 represents a handwritten figure
    _, axarr = plt.subplots(ncol, nrow)
    for ir in range(ncol):
        for ic in range(nrow):
            axarr[ir, ic].imshow(X[int(np.random.rand()*X.shape[0])].reshape((20,20)).T, cmap = cm.Greys_r)
            axarr[ir, ic].axis('off')
    plt.show()

handwritten_visualize(X, 10, 10)

def hw_demo(Xd, yd):
    train_set = []
    test_set = []
    train_set_size = 4950
    test_size = 50
    
    for idx in range(train_set_size):
        x = Xd[idx].reshape((-1,1))

        y = np.eye(10)[yd[idx]-1].reshape((-1, 1))

        train_set.append([x, y])

    # randomly choose some samles to test trained model
    for i in range(test_size):
        idx = int(np.random.rand() * len(Xd))
        x = Xd[idx].reshape((-1,1))
        y = np.eye(10)[yd[idx]-1].reshape((-1, 1))
        test_set.append([x, y])
    # create a network with two input, two hidden, and one output nodes
    n = NN(400, 25, 10)
    
    # train it with some patterns
    dt_st = datetime.datetime.now()
    print ("train start at: {0}".format(dt_st))
    
    n.train(train_set, 40, 10.1)
    
    dt_end = datetime.datetime.now()
    print ("train end at: {0}".format(dt_end))
    print ("time elapse in traininf: {0} seconds".format((dt_end-dt_st).seconds))
    
    n.test(test_set)

hw_demo(X, y)

                