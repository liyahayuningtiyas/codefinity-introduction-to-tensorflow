import tensorflow as tf
import numpy as np
import pandas as pd

# 1. Tensor initialization
tensor_A = tf.fill((3,3), 5)
tensor_B = tf.Variable([[2,3,4],[5,6,7]])
tensor_C = tf.zeros((3,3))
tensor_D = tf.ones((4,4))
tensor_E = tf.linspace(3,15,5)
tensor_F = tf.random.normal((2,2))

# Set up a NumPy array and a DataFrame
np_array = np.array([[7, 8], [9, 10], [11, 12]])
df = pd.DataFrame({'col1': [13, 14, 15], 'col2': [16, 17, 18]})

# 2. Conversions
tensor_from_array = tf.convert_to_tensor(np_array)
tensor_from_dataframe = tf.convert_to_tensor(df.values)

# Print out each tensor
print('tensor_A:', tensor_A, '\n', '-' * 80)
print('tensor_B:', tensor_B, '\n', '-' * 80)
print('tensor_C:', tensor_C, '\n', '-' * 80)
print('tensor_D:', tensor_D, '\n', '-' * 80)
print('tensor_E:', tensor_E, '\n', '-' * 80)
print('tensor_F:', tensor_F, '\n', '-' * 80)
print('tensor_from_array:', tensor_from_array, '\n', '-' * 80)
print('tensor_from_dataframe:', tensor_from_dataframe)