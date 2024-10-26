#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NUM_POINTS 1031      // Number of data points (adjust as needed)
#define NUM_FEATURES 8    // Number of features

float predict(float *theta, float *data_point, int num_features) {
    float prediction = 0.0;
    for (int i = 0; i < num_features; i++) {
        prediction += theta[i] * data_point[i];
    }
    return prediction;
}

void readCSV(const char *filename, float *X, float *y, int N, int d) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Could not open file %s\n", filename);
        exit(1);
    }

    // Skip header line
    char line[1024];
    fgets(line, sizeof(line), file);

    for (int i = 0; i < N; i++) {
        if (fgets(line, sizeof(line), file)) {
            // Split the line into tokens based on commas
            char *token = strtok(line, ",");
            int feature_index = 0;

            // Parse features (first 8 columns)
            while (token != NULL && feature_index < d) {
                float value = atof(token);  // Convert token to float
                X[i * d + feature_index] = value;
                token = strtok(NULL, ",");  // Get next token
                feature_index++;
            }

            // Parse target (last column: compressive strength)
            if (token != NULL) {
                float target_value = atof(token);  // Convert target to float
                y[i] = target_value;
            } else {
                printf("Error: Missing target value in row %d\n", i + 1);
            }
        } else {
            printf("Error: Could not read row %d\n", i + 1);
        }
    }

    fclose(file);
}

// CUDA Kernel for computing cost and gradient
__global__ void computeCostAndGradient(float *X, float *y, float *theta, float *cost, float *gradient, int m, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < m) {
        float h = 0.0; // Hypothesis
        // Calculate h(theta) = theta^T * x
        for (int j = 0; j < d; j++) {
            h += theta[j] * X[idx * d + j];
        }

        // Calculate error
        float error = h - y[idx];

        // Update cost (MSE)
        atomicAdd(cost, error * error); // Accumulate the squared error

        // Update gradient
        for (int j = 0; j < d; j++) {
            atomicAdd(&gradient[j], error * X[idx * d + j]); // Accumulate gradient
        }
    }
}



void linearRegression(float *h_X, float *h_y, float *h_theta, int m, int d) {
    // Parameters
    const float learning_rate = 0.0000001; // Adjust based on your data
    const int num_epochs = 1000000; // Number of iterations

    // Device memory allocation
    float *d_X, *d_y, *d_theta, *d_cost, *d_gradient;
    cudaMalloc((void **)&d_X, m * d * sizeof(float));
    cudaMalloc((void **)&d_y, m * sizeof(float));
    cudaMalloc((void **)&d_theta, d * sizeof(float));
    cudaMalloc((void **)&d_cost, sizeof(float));
    cudaMalloc((void **)&d_gradient, d * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_X, h_X, m * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta, h_theta, d * sizeof(float), cudaMemcpyHostToDevice);

    // Host variables for cost and gradient
    float h_cost;
    float *h_gradient = (float *)malloc(d * sizeof(float));

    // Training loop
    int blockSize = 512;
    int numBlocks = (m + blockSize - 1) / blockSize;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Initialize cost and gradient to zero on device
        cudaMemset(d_cost, 0, sizeof(float));
        cudaMemset(d_gradient, 0, d * sizeof(float));

        // Run kernel to compute cost and gradient
        computeCostAndGradient<<<numBlocks, blockSize>>>(d_X, d_y, d_theta, d_cost, d_gradient, m, d);
        cudaDeviceSynchronize();

        // Copy cost and gradient back to host
        cudaMemcpy(&h_cost, d_cost, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_gradient, d_gradient, d * sizeof(float), cudaMemcpyDeviceToHost);

        // Gradient descent update
        for (int j = 0; j < d; j++) {
            h_theta[j] -= learning_rate * h_gradient[j] / m;
        }

        // Copy updated theta back to device for the next iteration
        cudaMemcpy(d_theta, h_theta, d * sizeof(float), cudaMemcpyHostToDevice);

        // Optionally, print the cost every 100 epochs for monitoring
        if (epoch % 100000 == 0) {
            printf("Epoch %d, Cost: %f\n", epoch, h_cost / (2 * m));
        }
    }

    // Copy final theta values from device to host
    cudaMemcpy(h_theta, d_theta, d * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_X);
    cudaFree(d_y);
    cudaFree(d_theta);
    cudaFree(d_cost);
    cudaFree(d_gradient);
    free(h_gradient);
}


int main(int argc, char *argv[]) {
    // Assuming the CSV reading function is already implemented
    const char *filename = argv[1];  // Get CSV filename from command-line arguments

    // Host memory allocation for features and target
    float *h_X = (float *)malloc(NUM_POINTS * NUM_FEATURES * sizeof(float));  // NUM_POINTS x NUM_FEATURES matrix
    float *h_y = (float *)malloc(NUM_POINTS * sizeof(float));      // Target vector

    // Read CSV data into host memory
    readCSV(filename, h_X, h_y, NUM_POINTS, NUM_FEATURES);

    // Initialize theta (parameters) to zero
    float *h_theta = (float *)malloc(NUM_FEATURES * sizeof(float));
    for (int i = 0; i < NUM_FEATURES; i++) {
        h_theta[i] = ((float)rand() / RAND_MAX) * 0.01;
    }

    // Perform linear regression to calculate cost and gradient
    linearRegression(h_X, h_y, h_theta, NUM_POINTS, NUM_FEATURES);

    // Display learned theta values
    printf("Learned theta values:\n");
    for (int i = 0; i < NUM_FEATURES; i++) {
        printf("Theta[%d]: %f\n", i, h_theta[i]);
    }

    // Define the data point for prediction
    float data_point[NUM_FEATURES] = {332.5 ,142.5 ,0.0 ,228.0 ,0.0 ,932.0 ,594.0 ,270};

    // Make a prediction
    float predicted_value = predict(h_theta, data_point, NUM_FEATURES);
    printf("Predicted value: %f\n", predicted_value);


    // Free host memory
    free(h_X);
    free(h_y);
    free(h_theta);

    return 0;
}
