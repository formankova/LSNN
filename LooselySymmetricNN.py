import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

class LooselySymmetricNN():
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
        if not self.is_Fitted:
            raise Exception("Model has not been fitted yet")

    def _check_init(self):
        if self.enhancement_type not in self.enhancement_functions:
            raise Exception(
                "enhancement_type key not valid, supported options are none, save_node_unified, value_node_unified, save_node_unified_flattened, value_node_unified_flattened")

    def _check_input(self, X):
        if type(X) is not np.ndarray:
            raise ValueError("Array of a type np.nadarray expected")

        if (X.ndim != 2 or X.shape[1] != self.n_input):
            raise ValueError("Array of a shape (count_data, " + str(self.n_input) + ") expected")

    def _check_fit(self, X, y):
        self._check_input(X)
        if type(y) is not np.ndarray:
            raise ValueError("Array of a type np.nadarray expected")

        if y.shape != ((X.shape[0],)):
            raise ValueError("Incorrect expected output shape.")

    def _check_weights(self, wh, wo):
        if wh.shape != self.w_h.shape or type(wh) is not np.ndarray:
            raise ValueError("Array of a shape " + str(self.w_h.shape) + " expected")
        if wo.shape != self.w_o.shape or type(wo) is not np.ndarray:
            raise ValueError("Array of a shape " + str(self.w_o.shape) + " expected")

    def fit(self, X_train, y_train):
        self._check_fit(X_train, y_train)

        X, y = shuffle(X_train, y_train)

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
        # backpropagation - adjusting output layer weights
        adj_o = np.zeros((self.n_hidden, self.n_output))

        delta_output = self._delta_output(a3, target)
        ls = self._loosely_symmetric(a2, z3)
        n_a2 = np.where(a2 < ls, a2 * (1.0 + self.enhancement), a2)
        n_a2 = np.where(a2 > ls, a2 * (1.0 - self.enhancement), n_a2)

        if (save):
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
            if (save):
                a1 = n_a1
            adj_h[index] = - self.alpha * delta_hidden * n_a1

        self.w_o += adj_o.reshape(self.w_o.shape)
        self.w_h += adj_h.T

    def _adjustment_none_unified_flattened(self, a1, a2, a3, z2, z3, target, save):
        # backpropagation - adjusting output layer weights
        adj_o = np.zeros((self.n_hidden, self.n_output))

        delta_output = self._delta_output(a3, target)
        ls = self._loosely_symmetric(a2, z3)

        n_a2 = np.where(a2 < ls, np.where(a2 * (self.enhancement) < np.abs(ls - a2), a2 * (1.0 + self.enhancement), a2),
                        a2)
        n_a2 = np.where(a2 < ls, np.where(a2 * (self.enhancement) > np.abs(ls - a2), ls, n_a2), n_a2)
        n_a2 = np.where(a2 > ls,
                        np.where(a2 * (self.enhancement) < np.abs(ls - a2), a2 * (1.0 - self.enhancement), n_a2), n_a2)
        n_a2 = np.where(a2 > ls, np.where(a2 * (self.enhancement) > np.abs(ls - a2), ls, n_a2), n_a2)
        if (save):
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
                            np.where(a1 * (self.enhancement) < np.abs(ls - a1), a1 * (1.0 + self.enhancement), a1), a1)
            n_a1 = np.where(a1 < ls, np.where(a1 * (self.enhancement) > np.abs(ls - a1), ls, n_a1), n_a1)
            n_a1 = np.where(a1 > ls,
                            np.where(a1 * (self.enhancement) < np.abs(ls - a1), a1 * (1.0 - self.enhancement), n_a1),
                            n_a1)
            n_a1 = np.where(a1 > ls, np.where(a1 * (self.enhancement) > np.abs(ls - a1), ls, n_a1), n_a1)
            if (save):
                a1 = n_a1
            adj_h[index] = - self.alpha * delta_hidden * n_a1

        self.w_o += adj_o.reshape(self.w_o.shape)
        self.w_h += adj_h.T

    def _adjustment_save_node_unified(self, a1, a2, a3, z2, z3, target):
        self._adjustment_none_unified(a1, a2, a3, z2, z3, target, True)

    def _adjustment_value_node_unified(self, a1, a2, a3, z2, z3, target):
        self._adjustment_none_unified(a1, a2, a3, z2, z3, target, False)

    def _adjustment_save_node_unified_flattened(self, a1, a2, a3, z2, z3, target):
        self._adjustment_none_unified_flattened(a1, a2, a3, z2, z3, target, True)

    def _adjustment_value_node_unified_flattened(self, a1, a2, a3, z2, z3, target):
        self._adjustment_none_unified_flattened(a1, a2, a3, z2, z3, target, False)

    def _adjustment_enh_none(self, a1, a2, a3, z2, z3, target):
        # backpropagation - adjusting output layer weights
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
        return -(target - output) * self._sigmoid_derivative(output)

    def _delta_hidden(self, delta_output_times_w_h, output):
        return self._sigmoid_derivative(output) * delta_output_times_w_h

    def _loosely_symmetric(self, a, d):
        b = np.ones((a.shape)) - a
        c = 1 - d

        bd = (b * d) / (b + d)
        ac = (a * c) / (a + c)
        return (a + bd) / (1 + ac + bd)

    def _predict_value(self, X):
        a1, a2, a3, z2, z3 = self._feedforward(X)
        return round(a3[0])

    def predict(self, X):
        self._check_input(X)
        self._check_predict()
        result = []
        for x in X:
            result.append(self._predict_value(x))
        return result

    def _initialize_weights(self):
        w1 = np.random.randn(self.n_input, self.n_hidden) / np.sqrt(self.n_hidden)
        w2 = np.random.randn(self.n_hidden, self.n_output) / np.sqrt(self.n_output)
        return w1, w2

    def _error_squared(self, actual, predicted):
        error = 0
        for i in range(len(actual)):
            error += (predicted[i] - actual[i]) ** 2
        return error * 0.5

    # Accuracy classification score
    def accuracy_score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    # Weighted average of the precision and recall
    def f1_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average='binary')

    # Measures the ability of the classifier to find all the negative samples
    def precision_score(self, y_true, y_pred):
        return precision_score(y_true, y_pred, average='binary')

    # Measures the ability of the classifier to find all the positive samples
    def recall_score(self, y_true, y_pred):
        return recall_score(y_true, y_pred, average='binary')

    def eval(self, X, y):
        self._check_fit(X, y)
        result = []
        for x in X:
            result.append(self._predict_value(x))
        accuracy = self.accuracy_score(y, result)
        f1_score = self.f1_score(y, result)
        precision_score = self.precision_score(y, result)
        recall_score = self.recall_score(y, result)
        return accuracy, f1_score, precision_score, recall_score

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def get_weights(self):
        return self.w_h, self.w_o

    def set_weights(self, w_h, w_o):
        self._check_weights(w_h, w_o)
        self.w_h = w_h
        self.w_o = w_o
