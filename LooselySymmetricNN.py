import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

"""
Three-layered Loosely Symmetric Neural Network
"""


class LooselySymmetricNN:
    def __init__(self, n_input, n_hidden=30,
                 epochs=100, alpha=0.5,
                 random_state=1, enhancement=0.1, enhancement_type="none"):

        np.random.seed(random_state)  # for weights initialization

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = 1  # binary classifier

        self.w_h, self.w_o = self._initialize_weights()

        self.epochs = epochs + 1
        self.alpha = alpha

        self.enhancement = enhancement

        self.enhancement_type = enhancement_type

        self.enhancement_functions = {
            "none": self._adjustment_enh_none,
            "save_node_unified": self._adjustment_save_node_unified,
            "value_node_unified": self._adjustment_value_node_unified,
            "save_node_unified_flattened": self._adjustment_save_node_unified_flattened,
            "value_node_unified_flattened": self._adjustment_value_node_unified_flattened
        }

        self._check_init()

        self.is_Fitted = False

    def _check_predict(self):
        """
        Raise error when classifier is not trained.
        """
        if not self.is_Fitted:
            raise Exception("Model has not been fitted yet")

    def _check_init(self):
        """
        Raises error when selected enhancement type is not valid.
        """
        if self.enhancement_type not in self.enhancement_functions:
            raise Exception(
                "enhancement_type key not valid, supported options are none, save_node_unified, value_node_unified, "
                "save_node_unified_flattened, value_node_unified_flattened")

    def _check_input(self, X):
        """
        Raises error when X does not have the correct type or the valid shape.
        :param X: the array of inputs to neural network
        """
        if type(X) is not np.ndarray:
            raise ValueError("Array of a type np.nadarray expected")

        if X.ndim != 2 or X.shape[1] != self.n_input:
            raise ValueError("Array of a shape (count_data, " + str(self.n_input) + ") expected")

    def _check_fit(self, X, y):
        """
        Raises error when X or y does not have the correct type or the valid shape.
        :param X: the array of inputs
        :param y: the array of expected outputs
        """
        self._check_input(X)
        if type(y) is not np.ndarray:
            raise ValueError("Array of a type np.nadarray expected")

        if y.shape != (X.shape[0],):
            raise ValueError("Incorrect expected output shape.")

    def _check_weights(self, wh, wo):
        """
        It is used to check custom weights that are passed to the model.
        :param wh: weights between input and hidden layer
        :param wo: weights between hidden and output layer
        """
        if wh.shape != self.w_h.shape or type(wh) is not np.ndarray:
            raise ValueError("Array of a shape " + str(self.w_h.shape) + " expected")
        if wo.shape != self.w_o.shape or type(wo) is not np.ndarray:
            raise ValueError("Array of a shape " + str(self.w_o.shape) + " expected")

    def fit(self, X_train, y_train):
        """
        Fit function - the model's training process
        :param X_train: array of training examples
        :param y_train: array of expected labels
        :return: self
        """
        self._check_fit(X_train, y_train)

        X_train, y_train = shuffle(X_train, y_train)

        for i in range(self.epochs):

            # iterate over training set
            for j in range(X_train.shape[0]):
                # target for actual input
                target = y_train[j]

                # trained input
                X = X_train[j]

                # activations
                a1, a2, a3, z2, z3 = self._feedforward(X)

                self.enhancement_functions[self.enhancement_type](a1, a2, a3, z2, z3, target)

        self.is_Fitted = True
        return self

    def _adjustment_none_unified(self, a1, a2, a3, z2, z3, target, save):
        """
        Backpropagation for enhancement type '_node_unified'. Updates weights.
        :param a1: input
        :param a2: activation 2
        :param a3: activation 3
        :param z2: w_h*a1
        :param z3: w_o*a2
        :param target: expected labels
        :param save: boolean
        """
        # adjusting output layer weights
        adj_o = np.zeros((self.n_hidden, self.n_output))

        delta_output = self._delta_output(a3, target)
        ls = self._loosely_symmetric(a2, z3)
        n_a2 = np.where(a2 < ls, a2 * (1.0 + self.enhancement), a2)
        n_a2 = np.where(a2 > ls, a2 * (1.0 - self.enhancement), n_a2)

        if save:
            a2 = n_a2
        adj_o = - self.alpha * delta_output * n_a2

        adj_h = np.zeros((self.n_hidden, self.n_input))

        # for each hidden node
        for index in range(self.n_hidden):
            delta_output_times_w_h = self.w_o[index] * self._delta_output(
                a3, target)
            delta_hidden = self._delta_hidden(
                delta_output_times_w_h, a2[index])

            ls = self._loosely_symmetric(a1, z2[index])
            n_a1 = np.where(a1 < ls, a1 * (1.0 + self.enhancement), a1)
            n_a1 = np.where(a1 > ls, a1 * (1.0 - self.enhancement), n_a1)
            if save:
                a1 = n_a1
            adj_h[index] = - self.alpha * delta_hidden * n_a1

        self.w_o += adj_o.reshape(self.w_o.shape)
        self.w_h += adj_h.T

    def _adjustment_none_unified_flattened(self, a1, a2, a3, z2, z3, target, save):
        """
        Backpropagation for enhancement type '_node_unified_flattened'. Updates weights.
        :param a1: input
        :param a2: activation 2
        :param a3: activation 3
        :param z2: w_h*a1
        :param z3: w_o*a2
        :param target: expected labels
        :param save: boolean
        """
        # adjusting output layer weights
        adj_o = np.zeros((self.n_hidden, self.n_output))

        delta_output = self._delta_output(a3, target)
        ls = self._loosely_symmetric(a2, z3)

        n_a2 = np.where(a2 < ls, np.where(a2 * self.enhancement < np.abs(ls - a2), a2 * (1.0 + self.enhancement), a2),
                        a2)
        n_a2 = np.where(a2 < ls, np.where(a2 * self.enhancement > np.abs(ls - a2), ls, n_a2), n_a2)
        n_a2 = np.where(a2 > ls,
                        np.where(a2 * self.enhancement < np.abs(ls - a2), a2 * (1.0 - self.enhancement), n_a2), n_a2)
        n_a2 = np.where(a2 > ls, np.where(a2 * self.enhancement > np.abs(ls - a2), ls, n_a2), n_a2)
        if save:
            a2 = n_a2
        adj_o = - self.alpha * delta_output * n_a2

        adj_h = np.zeros((self.n_hidden, self.n_input))

        # for each hidden node
        for index in range(self.n_hidden):
            delta_output_times_w_h = self.w_o[index] * self._delta_output(
                a3, target)
            delta_hidden = self._delta_hidden(
                delta_output_times_w_h, a2[index])

            ls = self._loosely_symmetric(a1, z2[index])
            n_a1 = np.where(a1 < ls,
                            np.where(a1 * self.enhancement < np.abs(ls - a1), a1 * (1.0 + self.enhancement), a1), a1)
            n_a1 = np.where(a1 < ls, np.where(a1 * self.enhancement > np.abs(ls - a1), ls, n_a1), n_a1)
            n_a1 = np.where(a1 > ls,
                            np.where(a1 * self.enhancement < np.abs(ls - a1), a1 * (1.0 - self.enhancement), n_a1),
                            n_a1)
            n_a1 = np.where(a1 > ls, np.where(a1 * self.enhancement > np.abs(ls - a1), ls, n_a1), n_a1)
            if save:
                a1 = n_a1
            adj_h[index] = - self.alpha * delta_hidden * n_a1

        self.w_o += adj_o.reshape(self.w_o.shape)
        self.w_h += adj_h.T

    def _adjustment_save_node_unified(self, a1, a2, a3, z2, z3, target):
        """
        Executes self._adjustment_node_unified with param target=True
        :param a1: input
        :param a2: activation 2
        :param a3: activation 3
        :param z2: w_h*a1
        :param z3: w_o*a2
        :param target: expected labels
        """
        self._adjustment_none_unified(a1, a2, a3, z2, z3, target, True)

    def _adjustment_value_node_unified(self, a1, a2, a3, z2, z3, target):
        """
        Executes self._adjustment_node_unified with param target=False
        :param a1: input
        :param a2: activation 2
        :param a3: activation 3
        :param z2: w_h*a1
        :param z3: w_o*a2
        :param target: expected labels
        """
        self._adjustment_none_unified(a1, a2, a3, z2, z3, target, False)

    def _adjustment_save_node_unified_flattened(self, a1, a2, a3, z2, z3, target):
        """
        Executes self._adjustment_node_unified_flattened with param target=True
        :param a1: input
        :param a2: activation 2
        :param a3: activation 3
        :param z2: w_h*a1
        :param z3: w_o*a2
        :param target: expected labels
        """
        self._adjustment_none_unified_flattened(a1, a2, a3, z2, z3, target, True)

    def _adjustment_value_node_unified_flattened(self, a1, a2, a3, z2, z3, target):
        """
        Executes self._adjustment_node_unified_flattened with param target=False
        :param a1: input
        :param a2: activation 2
        :param a3: activation 3
        :param z2: w_h*a1
        :param z3: w_o*a2
        :param target: expected labels
        """
        self._adjustment_none_unified_flattened(a1, a2, a3, z2, z3, target, False)

    def _adjustment_enh_none(self, a1, a2, a3, z2, z3, target):
        """
        Backpropagation without LS model usage.
        :param a1: input
        :param a2: activation 2
        :param a3: activation 3
        :param z2: w_h*a1
        :param z3: w_o*a2
        :param target: expected labels
        """
        # adjusting output layer weights
        adj_o = np.zeros((self.n_hidden, self.n_output))

        delta_output = self._delta_output(a3, target)
        adj_o = - self.alpha * delta_output * a2

        adj_h = np.zeros((self.n_hidden, self.n_input))

        # for each hidden node
        for index in range(self.n_hidden):
            delta_output_times_w_h = self.w_o[index] * self._delta_output(
                a3, target)
            delta_hidden = self._delta_hidden(
                delta_output_times_w_h, a2[index])

            adj_h[index] = - self.alpha * delta_hidden * a1

        self.w_o += adj_o.reshape(self.w_o.shape)
        self.w_h += adj_h.T

    def _feedforward(self, X):
        """
        Feedforward process
        """
        # input
        a1 = X.astype(np.float)

        # wieghted sum - hidden layer
        z2 = a1.dot(self.w_h)
        a2 = self._sigmoid(z2.astype(np.float))

        # wieghted sum - output layer
        z3 = a2.dot(self.w_o)
        a3 = self._sigmoid(z3.astype(np.float))
        return a1, a2, a3, z2, z3

    def _delta_output(self, output, target):
        """
        Product of error and sigmoid derivative of output.
        :param output: real output
        :param target: expected output
        """
        return -(target - output) * self._sigmoid_derivative(output)

    def _delta_hidden(self, delta_output_times_w_h, output):
        """
        Product of the backpropagated delta and the sigmoid derivative of activation a2.
        :param delta_output_times_w_h:
        :param output: a2
        """
        return self._sigmoid_derivative(output) * delta_output_times_w_h

    @staticmethod
    def _loosely_symmetric(a, d):
        """
        Loosely symmetric model equation.
        """
        b = np.ones(a.shape) - a
        c = 1 - d

        bd = (b * d) / (b + d)
        ac = (a * c) / (a + c)
        return (a + bd) / (1 + ac + bd)

    def _predict_value(self, X):
        """
        Predicts input's label.
        :param X: input
        :return: 0 or 1
        """
        a1, a2, a3, z2, z3 = self._feedforward(X)
        return round(a3[0])

    def predict(self, X):
        """
        Predicts label.
        :param X: inputs
        :return: predicted labels
        """
        self._check_input(X)
        self._check_predict()
        result = []
        for x in X:
            result.append(self._predict_value(x))
        return result

    def _initialize_weights(self):
        """
        Weights initialization called at the model initialization.
        """
        w1 = np.random.randn(self.n_input, self.n_hidden) / np.sqrt(self.n_hidden)
        w2 = np.random.randn(self.n_hidden, self.n_output) / np.sqrt(self.n_output)
        return w1, w2

    @staticmethod
    def _error_squared(actual, predicted):
        """
        Measures the sum-of-squares error.
        :param actual: real output
        :param predicted: expected output
        :return: error
        """
        error = 0
        for i in range(len(actual)):
            error += (predicted[i] - actual[i]) ** 2
        return error * 0.5

    @staticmethod
    def accuracy_score(y_true, y_pred):
        """
        Measures accuracy score.
        :param y_true: real output
        :param y_pred: expected output
        :return: accuracy score
        """
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        Measures F-Measure - weighted average of the precision and recall.
        :param y_true: real output
        :param y_pred: expected output
        :return: F-Measure
        """
        return f1_score(y_true, y_pred, average='binary')

    @staticmethod
    def precision_score(y_true, y_pred):
        """
        Measures precision score - the ability of the classifier to find all the negative samples.
        :param y_true: real output
        :param y_pred: expected output
        :return: precision score
        """
        return precision_score(y_true, y_pred, average='binary')

    @staticmethod
    def recall_score(y_true, y_pred):
        """
        Measures recall score - the ability of the classifier to find all the positive samples.
        :param y_true: real output
        :param y_pred: expected output
        :return: recall score
        """
        return recall_score(y_true, y_pred, average='binary')

    def eval(self, X, y):
        """
        Predicts samples and returns accuracy, f-measure, precision and recall score.
        :param X: input
        :param y: expected output
        :return: accuracy, f-measure, precision and recall score
        """
        self._check_fit(X, y)
        result = []
        for x in X:
            result.append(self._predict_value(x))
        accuracy = self.accuracy_score(y, result)
        f1_score = self.f1_score(y, result)
        precision_score = self.precision_score(y, result)
        recall_score = self.recall_score(y, result)
        return accuracy, f1_score, precision_score, recall_score

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _sigmoid_derivative(x):
        return x * (1 - x)

    def get_weights(self):
        """
        Getter for trained weights.
        :return: w_h, w_o
        """
        return self.w_h, self.w_o

    def set_weights(self, w_h, w_o):
        """
        Setter for weights.
        :param w_h: weights between input and hidden layer
        :param w_o: weights between hidden and output layer
        """
        self._check_weights(w_h, w_o)
        self.w_h = w_h
        self.w_o = w_o
