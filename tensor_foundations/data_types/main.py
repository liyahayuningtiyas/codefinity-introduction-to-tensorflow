import tensorflow as tf

# 1. Create the initial tensor
initial_tensor = tf.random.normal((3,3), dtype=tf.float64)

# 2. Convert the tensor's data type
converted_tensor = tf.cast(initial_tensor,dtype=tf.float16)

# Display tensors
print("Initial Tensor:")
print(initial_tensor)
print('-' * 80)
print("Converted Tensor:")
print(converted_tensor)