# CUDA Linear Regression

## Overview

This project implements linear regression using CUDA for efficient computation. It reads a dataset from a CSV file and performs linear regression to learn the parameters (theta) using gradient descent. The implementation leverages parallel computing capabilities of CUDA to accelerate the computation of cost and gradient, which is essential for training the model.

## Kernels

### CUDA Kernel: `computeCostAndGradient`

The primary CUDA kernel used in this implementation is `computeCostAndGradient`. This kernel performs the following tasks in parallel for each data point:

1. **Hypothesis Calculation**: Computes the predicted value (\(h\)) based on the current parameters (\(\theta\)) and the input features (\(X\)).
2. **Error Calculation**: Calculates the error by subtracting the actual target value (\(y\)) from the predicted value.
3. **Cost Accumulation**: Updates the total cost using Mean Squared Error (MSE) via atomic operations to ensure thread safety.
4. **Gradient Accumulation**: Computes the gradient for each feature and accumulates these values using atomic operations.

### Algorithm

The overall algorithm follows these steps:

1. **Data Reading**: The program reads feature values and target values from a CSV file into host memory.
2. **Parameter Initialization**: Initializes the parameter vector (\(\theta\)) to small random values.
3. **Training Loop**:
   - For a specified number of epochs:
     - Reset cost and gradient on the device.
     - Launch the `computeCostAndGradient` kernel to compute the cost and gradient.
     - Copy the cost and gradient back to host memory.
     - Update the parameter vector (\(\theta\)) using gradient descent.
     - Print the cost every 100,000 epochs for monitoring.
4. **Prediction**: After training, the learned parameters are used to predict values for new data points.

## How to Run

1. **Prerequisites**: Ensure you have a CUDA-capable GPU and the CUDA toolkit installed.

2. **Compile the Code**: Use the following command to compile the code:
   ```bash
   nvcc -o linear_regression linear_regression.cu

