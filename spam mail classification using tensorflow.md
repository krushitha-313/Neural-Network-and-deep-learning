import tensorflow as tf

# Weights and biases
W1 = tf.constant([
    [0.5, -0.2, 0.1, 0.3],
    [0.8, 0.4, -0.6, 0.2],
    [0.1, 0.9, 0.5, -0.7]
], dtype=tf.float32)

B1 = tf.constant([0.1, -0.5, 0.2, 0.0], dtype=tf.float32)

W2 = tf.constant([
    [0.9],
    [-0.3],
    [0.7],
    [0.2]
], dtype=tf.float32)

B2 = tf.constant([0.5], dtype=tf.float32)

# Forward pass
def forward_pass(inputs):
    Z1 = tf.matmul(inputs, W1) + B1         # Hidden layer weighted sum
    A1 = tf.nn.relu(Z1)                     # ReLU activation
    Z2 = tf.matmul(A1, W2) + B2             # Output layer weighted sum
    A2 = tf.nn.sigmoid(Z2)                  # Sigmoid activation
    return Z1, A1, Z2, A2

# Correct test input shape
test_input = tf.constant([[1.0, 1.0, 1.0]], dtype=tf.float32)

Z1, A1, Z2, A2 = forward_pass(test_input)
probability = A2.numpy()[0][0]

print("Probability:", probability)
print("Prediction:", "SPAM" if probability > 0.5 else "NOT SPAM")
