# LooselySymmetricNN (LSNN)
Loosely Symmetric Neural Network Classifier.

Implementation of the neural network with enhancement by symmetric biases. It is based on the description by Taniguchi et al. (https://doi.org/10.9746/jcmsi.12.56).

## Requirements
+ numpy
+ sklearn.utils
+ sklearn.metrics

| Parameters    |                                       |       |
| ------------- |---------------------------------------|-------|
|               | **n_input**: int                          | number of features |
|               | **n_hidden**: int, default=30             | number fo nodes in the hidden layer |
|               | **epochs**: int, default=100              | training epochs |
|               | **alpha**: int, default=0.5               | learning rate |
|               | **random_state**: int, default=1          | used for weights initialization
|               | **enhancement**: float, default=0.1       | determines the gravity of the change made to nodes during backpropagation, ignored when the enhancement_type is 'none'
|               | **enhancement_type**: str, default="none" | |
|               | **shuffle**: bool, default=False | option to shuffle data before training |
| **Methods**       |  | |
|               | fit(X_train, y_train) | Fit the model to data matrix X and target y. |
|               | predict(X) | Predict target. |
|               | accuracy_score(y_true, y_pred) | Return accuracy score. |
|               | f1_score(y_true, y_pred)| Return F-measure. |
|               | precision_score(y_true, y_pred) | Return precision score. |
|               | recall_score(y_true, y_pred)| Return recall score. |
|               | eval(X, y) | Predict y and return accuracy score, F-measure, precision score and recall score. |
|               | get_weights() | Return net's weights |
|               | set_weights(w_h, w_o) | Set weights with new values |

## Enhancement types
+ none

  Classical feedforward neural network without enhancement by the loosely symmetric model.

+ save_node_unified
  
  Enlarges or lessens nodes according to LS model output. New nodes are backpropagated further.
  
+ value_node_unified

  Enlarges or lessens nodes according to LS model output. New nodes are **not** backpropagated further.

+ save_node_unified_flattened

  Values are not changed when the enhancement is more significant than the difference between nodes and LS model output, but LS model output is used. New nodes are backpropagated further.

+ value_node_unified_flattened

  Values are not changed when the enhancement is more significant than the difference between nodes and LS model output, but LS model output is used. New nodes are **not** backpropagated further.
  
## Contents
+ utils - preprocessing function for text data sets

+ test - contains usage example

+ datasets - raw SpamAssassin public corpus without hard_ham (source: https://spamassassin.apache.org/old/publiccorpus/)

+ LSNN_enhancementExperiment - extracted code from the experimentation with multiple possible enhancements using the LS model
