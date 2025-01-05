import numpy as np
import matplotlib.pyplot as plt

# Define the Softplus function
def softplus(x):
    return np.log(1 + np.exp(x)) / (1 + np.log(1 + np.exp(x)))

# Generate input values
x = np.linspace(-10, 10, 1000)

# Compute the Softplus output
y = softplus(x)

# Plot the Softplus function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Softplus', color='blue')
plt.title('Softplus Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (softplus(x))')
plt.grid(True)
plt.legend()

plt.show()
