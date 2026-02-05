import tensorflow as tf

# 1. Define the matrices `A` and `B`
A = tf.constant([[2, 3, -1],
                 [4, 1, 2],
                 [-1, 2, 3]], dtype=tf.float64)

B = tf.constant([[1],
                 [2],
                 [3]], dtype=tf.float64)

# 2. Find the inverse of matrix `A` using the correct API
A_inv = tf.linalg.inv(A)

# 3. Compute the solution matrix `X` by multiplying the inverse by B
X = tf.matmul(A_inv, B)

# Extract the values of `x`, `y`, and `z`
x, y, z = X[:, 0]

# Display the results
print(f"x: {x:.2f}")
print(f"y: {y:.2f}")
print(f"z: {z:.2f}")