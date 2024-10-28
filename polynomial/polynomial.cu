#include <iostream>
#include <cmath>
#include <cstdlib> // For atoi

#define MAX_ITER 1000
// #define LEARNING_RATE 0.001f // Smaller step size for stability

__device__ float evaluate_polynomial(const float *coefficients, int degree, float x) {
    float result = 0.0f;
    float power_of_x = 1.0f;
    for (int i = 0; i <= degree; ++i) {
        result += coefficients[i] * power_of_x;
        power_of_x *= x;
    }
    return result;
}

__device__ float evaluate_derivative(const float *coefficients, int degree, float x) {
    float result = 0.0f;
    float power_of_x = 1.0f;
    for (int i = 1; i <= degree; ++i) {
        result += i * coefficients[i] * power_of_x;
        power_of_x *= x;
    }
    return result;
}

__global__ void find_minima(const float *coefficients, int degree, float *minima, float *initial_guesses, int num_guesses,float LEARNING_RATE) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_guesses) return;

    float x = initial_guesses[idx];
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        float grad = evaluate_derivative(coefficients, degree, x);
        x -= LEARNING_RATE * grad;
        if (fabs(grad) < 1e-6) break; // Stricter convergence tolerance
    }

    minima[idx] = evaluate_polynomial(coefficients, degree, x);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <degree> <learning_rate>" << std::endl;
        return 1;
    }

    // Read degree of polynomial from command-line argument
    float LEARNING_RATE = atof(argv[2]);
    int degree = atoi(argv[1]);
    if (degree < 1) {
        std::cerr << "Degree must be a positive integer." << std::endl;
        return 1;
    }

    // Define polynomial coefficients for higher degrees
    float h_coefficients[degree + 1];
    if (degree == 2) {
        h_coefficients[0] = 2.0f;  // Constant term
        h_coefficients[1] = -3.0f; // Linear term
        h_coefficients[2] = 1.0f;  // Quadratic term
    } else if (degree == 3) {
        h_coefficients[0] = 0.0f;  // Constant term
        h_coefficients[1] = 1.0f;  // x term
        h_coefficients[2] = -2.0f; // x^2 term
        h_coefficients[3] = 1.0f;  // x^3 term
    } else {
        // Initialize with smaller, alternating coefficients for higher degrees
        for (int i = 0; i <= degree; ++i) {
            h_coefficients[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        }
    }

    int num_guesses = 256; // Number of initial guesses (i.e., threads)

    // Allocate memory on the device
    float *d_coefficients, *d_minima, *d_initial_guesses;
    cudaMalloc(&d_coefficients, (degree + 1) * sizeof(float));
    cudaMalloc(&d_minima, num_guesses * sizeof(float));
    cudaMalloc(&d_initial_guesses, num_guesses * sizeof(float));

    // Copy coefficients to device
    cudaMemcpy(d_coefficients, h_coefficients, (degree + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize initial guesses in a narrower range and copy to device
    float h_initial_guesses[num_guesses];
    for (int i = 0; i < num_guesses; ++i) {
        h_initial_guesses[i] = -5.0f + (10.0f * i) / num_guesses; // Narrowed range
    }
    cudaMemcpy(d_initial_guesses, h_initial_guesses, num_guesses * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to find minima
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_guesses + threadsPerBlock - 1) / threadsPerBlock;
    find_minima<<<blocksPerGrid, threadsPerBlock>>>(d_coefficients, degree, d_minima, d_initial_guesses, num_guesses,LEARNING_RATE);

    // Copy results back to host
    float h_minima[num_guesses];
    cudaMemcpy(h_minima, d_minima, num_guesses * sizeof(float), cudaMemcpyDeviceToHost);

    // Find the smallest minimum from all threads
    float global_min = h_minima[0];
    for (int i = 1; i < num_guesses; ++i) {
        if (h_minima[i] < global_min) global_min = h_minima[i];
    }

    std::cout << "Approximate minimum value of the polynomial: " << global_min << std::endl;

    // Free device memory
    cudaFree(d_coefficients);
    cudaFree(d_minima);
    cudaFree(d_initial_guesses);

    return 0;
}
