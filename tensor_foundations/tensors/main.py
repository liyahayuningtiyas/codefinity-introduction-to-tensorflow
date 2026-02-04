import tensorflow as tf

# Create tensors
tensor_1D = tf.constant([1,2,3])
tensor_2D = tf.constant([[1,2,3], [4,5,6]])
tensor_3D = tf.constant([[[1,2,3], [4,5,6],[7,8,9]]])

# Display tensors
print(tensor_1D)
print('-' * 50)
print(tensor_2D)
print('-' * 50)
print(tensor_3D)