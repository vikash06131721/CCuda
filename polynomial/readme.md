# CUDA Polynomial Optimization

## About the Kernels

This project implements a CUDA-based optimization algorithm that uses gradient descent to find the local minima of polynomials of varying degrees. The key components of the implementation include:

- **Kernel Function**: The `find_minima` kernel computes the minimum values of the polynomial using multiple initial guesses in parallel. Each thread performs gradient descent to converge towards a local minimum.

- **Device Functions**: The helper functions `evaluate_polynomial` and `evaluate_derivative` compute the value of the polynomial and its derivative, respectively, for a given input `x`.

## Gradient Descent

Gradient descent is an iterative optimization algorithm used to minimize a function. It works by updating the current guess for the minimum in the direction of the negative gradient (i.e., the direction of steepest descent). The steps involved are:

1. **Initialization**: Start with an initial guess for the minimum value.
2. **Gradient Calculation**: Compute the derivative (gradient) of the polynomial at the current guess.
3. **Update Step**: Adjust the guess by subtracting a fraction of the gradient (scaled by a learning rate).
4. **Convergence Check**: Repeat the process until the change in the guess is sufficiently small or a maximum number of iterations is reached.

## About the Polynomial Being Used

The code supports polynomials of different degrees, defined by the user at runtime. The following polynomials are implemented:

- **Degree 2**: 
  \[
  f(x) = x^2 - 3x + 2
  \]
  - Coefficients: \( a_2 = 1.0, a_1 = -3.0, a_0 = 2.0 \)

- **Degree 3**: 
  \[
  f(x) = x^3 - 2x^2 + x
  \]
  - Coefficients: \( a_3 = 1.0, a_2 = -2.0, a_1 = 1.0, a_0 = 0.0 \)

- **Degree > 3**: 
  A polynomial with alternating coefficients, defined as:
  \[
  f(x) = 0.5x^n - 0.5x^{n-1} + 0.5x^{n-2} - 0.5x^{n-3} + \dots
  \]
  - Coefficients: Alternating values of \(0.5\) and \(-0.5\).

## How to Run

1. **Prerequisites**: Ensure you have a CUDA-capable GPU and the CUDA toolkit installed.

2. **Compile the Code**: Use the following command to compile the code:
   ```bash
   nvcc -o polynomial polynomial.cu
   ./polynomial 2 0.001

### Tip

Try varying learning rate with degrees? What can you make of it.
