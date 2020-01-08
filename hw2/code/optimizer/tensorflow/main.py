from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, validation_size = 0)
import tensorflow as tf
import argparse
from optimizers import createOptimizer
import pickle
import numpy as np

def main(args):
   # Network Parameters
   n_hidden_1 = 256 # 1st layer number of neurons
   n_hidden_2 = 256 # 2nd layer number of neurons
   n_input = 784 # MNIST data input (img shape: 28*28)
   n_classes = 10 # MNIST total classes (0-9 digits)
   
   # tf Graph input
   X = tf.placeholder("float", [None, n_input])
   Y = tf.placeholder("float", [None, n_classes])
   
   # Store layers weight & bias
   weights = {
   'h1': tf.Variable(tf.random_uniform([n_input, n_hidden_1], minval = -1./ np.sqrt(n_input), maxval = 1./ np.sqrt(n_input))),
   'h2': tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2], minval = -1./ np.sqrt(n_hidden_1), maxval = 1./ np.sqrt(n_hidden_1))),
   'out': tf.Variable(tf.random_uniform([n_hidden_2, n_classes], minval = -1./ np.sqrt(n_hidden_2), maxval = 1./ np.sqrt(n_hidden_2)))
   }
   biases = {
   'b1': tf.Variable(tf.random_uniform([n_hidden_1], minval = -1./ np.sqrt(n_input), maxval = 1./ np.sqrt(n_input))),
   'b2': tf.Variable(tf.random_uniform([n_hidden_2], minval = -1./ np.sqrt(n_hidden_1), maxval = 1./ np.sqrt(n_hidden_1))),
   'out': tf.Variable(tf.random_uniform([n_classes], minval = -1./ np.sqrt(n_hidden_2), maxval = 1./ np.sqrt(n_hidden_2)))
   }


   def multilayer_perceptron(x):
      # Hidden fully connected layer with 256 neurons
      layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
      # Hidden fully connected layer with 256 neurons
      layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
      # Output fully connected layer with a neuron for each class
      out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
      return out_layer

   # Construct model
   logits = multilayer_perceptron(X)

   # Define loss and optimizer
   loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits=logits, labels=Y))

   base_optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
   grads_and_vars = base_optimizer.compute_gradients(loss_op)
   gradients = [g for g, v in grads_and_vars]
   variables = [v for g, v in grads_and_vars]
   
   #Computes the updates for the selected gradients, and updates 
   #the gradient values.
   optimizer = createOptimizer(args)
   optimizer.update(gradients)
   
   # Applies the updates to the variables.
   updated_grads_and_vars = [(g, v) for g, v in zip(gradients, variables)]
   train_op = base_optimizer.apply_gradients(updated_grads_and_vars)
   
      
   # Calculates the accuracy.
   pred = tf.nn.softmax(logits)  # Apply softmax to logits
   correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
   accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
   
   # Initializing the variables
   init = tf.global_variables_initializer()

   with tf.Session() as sess:
      sess.run(init)
      # Training cycle
      losses = []
      accuracies = []
      for epoch in range(args.training_epochs):
         avg_cost = 0.
         total_batch = int(mnist.train.num_examples/args.batch_size)
         # Loop over all batches
         for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(args.batch_size)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y:batch_y})

            # Compute average loss
            avg_cost += c / total_batch
      
         # Computes the train accuracy.
         avg_accuracy = accuracy.eval({X: mnist.train.images, Y: mnist.train.labels})
         accuracies.append(avg_accuracy)
         losses.append(avg_cost)
         print("Epoch:", '%04d' % (epoch+1), "cost={:.4f}".format(avg_cost)+" accuracy={:.4f}".format(avg_accuracy))
      print("Optimization Finished!")

      # Saves the losses and accuracies
      if args.output is not None:
         pickle.dump({"train_loss":losses, "train_accuracy": accuracies},\
            open(args.output, "wb"))  
   
      # Tests the model.
      print("Test accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--optimizer', type=str, default="rmsprop", help='Name of the optimizer: sgd,\
        momentumsgd, adam, adagrad, rmsprop')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--training_epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of batch sizes')
    parser.add_argument('--delta', type=float, default=1e-8, help='The camping coefficient')
    parser.add_argument('--tau', default=0.9, help='Decaying parameter')
    parser.add_argument('--rho', type=float, default=0.9, help='momentum')
    parser.add_argument('--beta1', type=float, default=0.9, help='first order decaying parameter')
    parser.add_argument('--beta2', type=float, default=0.999, help='second order decaying parameter')
    parser.add_argument('--output', type=str, default=None, help='Output file to save training loss\
            and accuracy.')
    args = parser.parse_args()
    print('Running '+args.optimizer+' optimizer ...')
    main(args)


