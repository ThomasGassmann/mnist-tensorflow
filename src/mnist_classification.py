import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

training_steps = 5000
learning_rate = 0.025

# Load MNIST dataset
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Define input and result placeholder values
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

def add_layer(input_tensor, output_size):
    # Initialize weights with random values
    weights = tf.Variable(tf.truncated_normal((int(input_tensor.shape[1]), output_size)))
    bias = tf.Variable(tf.truncated_normal([output_size]))
    # Weight * input + bias with sigmoid
    output = tf.nn.sigmoid(tf.matmul(input_tensor, weights) + bias)
    return output

# First hidden layer
h1 = add_layer(x, 256)
# Second hidden layer
h2 = add_layer(h1, 128)

# Output layer
logits = add_layer(h2, 10)

prediction = session.run(predictor, {x: x_test})

# Compute cross entropy between actual values and outputs
loss_function = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
# Minimize the loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

for step in range(training_steps):
    x_batch, y_batch = mnist.train.next_batch(100)
    session.run(optimizer, {x: x_batch, y: y_batch})

# Evaluate accuracy
predictor = tf.nn.softmax(logits)
x_test, y_test = mnist.test.next_batch(10000)
prediction_vector = np.argmax(prediction, axis = 1)
accuracy = np.mean(np.argmax(y_test, axis = 1) == prediction_vector)
print(accuracy)
