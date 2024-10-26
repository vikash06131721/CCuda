# CCuda
A repository for ML algorithms implementation in C CUDA.

## Linear Regression

### Code
File: `linearReg.cu`

### Data
Data file: `data/data.csv`

### To Experiment

To compile and run the linear regression implementation:

```bash
nvcc -o linearRegression linearReg.cu
./linearRegression data/data.csv
```
###Implementation Details

1. CUDA Kernel for Cost and Gradient Computation

The computeCostAndGradient kernel calculates the hypothesis, cost, and gradient values for each data point in parallel.


__global__ void computeCostAndGradient(float *X, float *y, float *theta, float *cost, float *gradient, int m, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        float h = 0.0;
        for (int j = 0; j < d; j++) {
            h += theta[j] * X[idx * d + j];
        }
        float error = h - y[idx];
        atomicAdd(cost, error * error);
        for (int j = 0; j < d; j++) {
            atomicAdd(&gradient[j], error * X[idx * d + j]);
        }
    }
}

Hypothesis Calculation: Each thread computes h, the hypothesis value, for a single data point using the current theta values.

Atomic Operations: atomicAdd ensures safe accumulation of cost and gradient values across threads.

2. Kernel Launch Parameters

int blockSize = 512;

int numBlocks = (m + blockSize - 1) / blockSize;


blockSize: Number of threads per block, typically set to 512 for optimal performance.


numBlocks: Number of blocks needed to cover all m data points.

3.  Host-Device Memory Management

Data points (X), target values (y), and parameters (theta) are stored in device memory for parallel processing. Memory allocation and data transfer between host and device are handled as follows:

cudaMalloc((void **)&d_X, m * d * sizeof(float));
cudaMalloc((void **)&d_y, m * sizeof(float));
cudaMalloc((void **)&d_theta, d * sizeof(float));
cudaMalloc((void **)&d_cost, sizeof(float));
cudaMalloc((void **)&d_gradient, d * sizeof(float));

cudaMemcpy(d_X, h_X, m * d * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_y, h_y, m * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_theta, h_theta, d * sizeof(float), cudaMemcpyHostToDevice);
