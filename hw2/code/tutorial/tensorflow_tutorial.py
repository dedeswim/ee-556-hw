import pdb
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.ERROR)

input_dim = 784
classes = 10
hidden = 100
batch_size = 100
epochs = 15
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n_batches = int(mnist.train.num_examples / batch_size)


if __name__ == '__main__':
    x = tf.placeholder('float', shape=[None, input_dim])
    y = tf.placeholder('float', shape=[None, classes])

    W1 = tf.Variable(tf.random_normal([input_dim, hidden]))
    b1 = tf.Variable(tf.random_normal([hidden]))

    W2 = tf.Variable(tf.random_normal([hidden, classes]))
    b2 = tf.Variable(tf.random_normal([classes]))

    layer1 = tf.add(tf.matmul(x, W1), b1)
    activation = tf.nn.relu(layer1)
    layer2 = tf.add(tf.matmul(activation, W2), b2)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
       logits=layer2, labels=y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    grads_and_vars = optimizer.compute_gradients(loss)
    train = optimizer.apply_gradients(grads_and_vars)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init) # Initialize weights and biases
        for epoch in range(epochs):
            total_loss = 0.
            for i in range(n_batches):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                if i == 0:
                    print('batch size: ' + str(batch_x.shape[0]))
                    print('input dimension: ' + str(batch_x[0].shape))
                _, batch_loss = sess.run([train, loss], feed_dict={x:batch_x, y:batch_y})
                total_loss += batch_loss / n_batches
            print('average loss: ' + str(total_loss))
            print(n_batches)
            break
