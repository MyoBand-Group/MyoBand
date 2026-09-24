#include <iostream>
#include <cuda_runtime.h>

// 1. The GPU Kernel (Function that runs on the GPU)
// __global__ tells the compiler this function is called from CPU but runs on GPU.
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    // Calculate the unique global thread ID for this specific thread
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Ensure we don't look past the end of the array
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000000; // 1 million elements
    size_t size = N * sizeof(float);

    // 2. Allocate Host (CPU) Memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize CPU data
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 3. Allocate Device (GPU) Memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 4. Copy data from CPU to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 5. Define execution configuration (Threads and Blocks)
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 6. Launch the Kernel
    // The <<< >>> syntax passes the grid and block dimensions to the GPU
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 7. Copy results back from GPU to CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify results
    std::cout << "Result at index 0: " << h_C[0] << " (Expected: 3)" << std::endl;
    std::cout << "Result at index 999999: " << h_C[999999] << " (Expected: 3)" << std::endl;

    // 8. Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}