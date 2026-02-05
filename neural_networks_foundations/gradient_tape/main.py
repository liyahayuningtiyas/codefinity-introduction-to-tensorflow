import tensorflow as tf

# 1. Define the variable `x` at the point of interest
x = tf.Variable(2.0)

# 2. Use Gradient Tape to record the computation of the function `f(x)`
with tf.GradientTape() as tape:
    f = x ** 2 + 2 * x - 1

# 3. Calculate the gradient of `f(x)` at the specified point
df_dx = tape.gradient(f,x)

# Print the results
print("The value of f(x) at x = 2 is:", f.numpy())
print("The derivative of f(x) at x = 2 is:", df_dx.numpy())