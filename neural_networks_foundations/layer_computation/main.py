import tensorflow as tf

# Define layer weights, layer inputs and layer bias
W = tf.Variable([[0.5, 0.4, 0.7], [0.1, 0.9, 0.5]], dtype=tf.float64)  # Matrix 2x3 
I = tf.constant([[0.5, 0.6, 0.1]], dtype=tf.float64)  # Matrix 1x3
b = tf.Variable(0.1, dtype=tf.float64)

# 1. Transpose weight matrix
W = tf.transpose(W)

# 2. Compute the weighted sum
weighted_sum = tf.matmul(I, W)

# 3. Add bias
biased_sum = weighted_sum + b 

# 4. Apply the sigmoid activation function
neuron_output = tf.sigmoid(biased_sum)

# Display results
print('Transposed weight matrix:\n', W)
print('-' * 50)
print('Weighted sum:\n', weighted_sum)
print('-' * 50)
print('Biased sum:\n', biased_sum)
print('-' * 50)
print('Output of the neuron:\n', neuron_output)