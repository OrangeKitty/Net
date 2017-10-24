import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import math
import matplotlib.pyplot as plt


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32,[n_x,None])
    Y = tf.placeholder(tf.float32,[n_y,None])  
    return X, Y

def initialize_parameters_deep(layer_dims):    
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
      parameters['W' + str(l)] = tf.get_variable(name = 'W' + str(l),
                                                shape = [layer_dims[l],layer_dims[l-1]],
                                                initializer = tf.contrib.layers.xavier_initializer())
      parameters['b' + str(l)] = tf.get_variable('b' + str(l), [layer_dims[l],1], initializer = tf.zeros_initializer())
    return parameters

def forward_propagation(X, parameters):
  L = len(parameters)//2
  A_prev = X
  Z = {}
  for l in range(1,L):
    Z['Z' + str(l)] = tf.add(tf.matmul(parameters['W'+str(l)],A_prev),parameters['b'+str(l)])
    A_prev = tf.nn.relu(Z['Z' + str(l)])
  
  Z['Z' + str(L)] = tf.add(tf.matmul(parameters['W'+str(L)],A_prev),parameters['b'+str(L)])
    
  return Z['Z' + str(L)]


def compute_cost(Z3, Y):
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    
    return cost

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
  np.random.seed(seed)
  m = X.shape[1]
  mini_batches = []
  permutation = list(np.random.permutation(m))
  shuffled_X = X[:, permutation]
  shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m)) 
  num_complete_minibatches = math.floor(m/mini_batch_size)
  for k in range(0, num_complete_minibatches): 
    mini_batch_X = shuffled_X[:,mini_batch_size*k:mini_batch_size*(k+1)]
    mini_batch_Y = shuffled_Y[:,mini_batch_size*k:mini_batch_size*(k+1)]
    
    mini_batch = (mini_batch_X, mini_batch_Y)
    mini_batches.append(mini_batch)
    
  if m % mini_batch_size != 0:
    mini_batch_X = shuffled_X[:,mini_batch_size*num_complete_minibatches:]
    mini_batch_Y = shuffled_Y[:,mini_batch_size*num_complete_minibatches:]
    
    mini_batch = (mini_batch_X, mini_batch_Y)
    mini_batches.append(mini_batch)
    
  return mini_batches

def model(X_train, Y_train, X_test, Y_test,h_layers, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    
  ops.reset_default_graph()
  tf.set_random_seed(1)
  seed = 3
  (n_x, m) = X_train.shape
  n_y = Y_train.shape[0]
  costs = []
  
  X, Y = create_placeholders(n_x, n_y)
  layer_dims = [n_x]
  layer_dims.extend(h_layers)
  layer_dims.extend([n_y])
  
  parameters = initialize_parameters_deep(layer_dims)
  Z3 = forward_propagation(X, parameters)
  cost = compute_cost(Z3, Y)
  optimizer = tf.train.AdadeltaOptimizer(learning_rate = learning_rate).minimize(cost)
  
  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
      epoch_cost = 0.
      num_minibatches = int(m / minibatch_size)
      seed = seed + 1
      minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
      for minibatch in minibatches:
        (minibatch_X, minibatch_Y) = minibatch
        _ , minibatch_cost = sess.run([optimizer,cost], feed_dict={X: minibatch_X, Y: minibatch_Y}) 
        epoch_cost += minibatch_cost / num_minibatches
      if print_cost == True and epoch % 100 == 0:
        print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
      if print_cost == True and epoch % 5 == 0:
        costs.append(epoch_cost)
          
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    parameters = sess.run(parameters)
    print ("Parameters have been trained!")
    
    correct_prediction = tf.equal(tf.argmax(Z3,axis=0), tf.argmax(Y,axis=0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
    print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
    return parameters

#parameters = model(X_train, Y_train, X_test, Y_test,h_layers = [50,40,30,20,10],learning_rate = 0.0002,num_epochs = 1500)