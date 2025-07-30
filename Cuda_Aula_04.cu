// Cuda_Aula_04.cu
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cassert>
#include <chrono> // For timing CPU operations

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Macro to wrap CUDA API calls and check for errors
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
                  << ": " << cudaGetErrorString(err_) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Define the structure for a matrix
// This structure will be used on both the host (CPU) and device (GPU)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// CUDA Kernel for Matrix Multiplication (C = A * B)
// This function runs on the GPU.
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    // Each thread computes one element of the output matrix C.
    float Cvalue = 0.0f;

    // Calculate the row and column of the C element to be computed by this thread.
    // blockIdx, blockDim, and threadIdx are built-in CUDA variables.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread is within the bounds of the output matrix C.
    // This is important if the matrix dimensions are not perfect multiples of the block dimensions.
    if (row < C.height && col < C.width) {
        // Iterate through the elements of the row in A and the column in B
        // to compute the dot product.
        for (int k = 0; k < A.width; ++k) {
            // A.width must be equal to B.height for valid matrix multiplication.
            // M(row, col) = *(M.elements + row * M.width + col)
            Cvalue += A.elements[row * A.width + k] * B.elements[k * B.width + col];
        }

        // Store the computed value in the correct position in the output matrix C.
        C.elements[row * C.width + col] = Cvalue;
    }
}

// Host function to perform matrix multiplication on the CPU for verification
void MatMulCPU(const Matrix& A, const Matrix& B, Matrix& C) {
    for (int i = 0; i < C.height; ++i) {
        for (int j = 0; j < C.width; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A.width; ++k) {
                sum += A.elements[i * A.width + k] * B.elements[k * B.width + j];
            }
            C.elements[i * C.width + j] = sum;
        }
    }
}

// Host function to print a matrix (for debugging)
void PrintMatrix(const Matrix& M) {
    std::cout << "Matrix (" << M.height << "x" << M.width << "):" << std::endl;
    for (int i = 0; i < M.height; ++i) {
        for (int j = 0; j < M.width; ++j) {
            std::cout << M.elements[i * M.width + j] << " ";
        }
        std::cout << std::endl;
    }
}


// Main program entry point
int main(int argc, char** argv) {
    // --- 1. Setup and Initialization ---

    // Seed the random number generator
    srand(time(0));

    // Define matrix dimensions
    // For C = A * B, A.width must equal B.height
    const int A_HEIGHT = 256;
    const int A_WIDTH = 512;
    const int B_HEIGHT = 512;
    const int B_WIDTH = 128;

    // The resulting matrix C will have dimensions A.height x B.width
    const int C_HEIGHT = A_HEIGHT;
    const int C_WIDTH = B_WIDTH;

    // Define CUDA block and grid dimensions
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const int BLOCK_SIZE_X = 16;
    const int BLOCK_SIZE_Y = 16;
    float milliseconds = 0;

    // --- 2. Allocate Host (CPU) Memory ---
    std::cout << "Allocating host memory..." << std::endl;
    auto start_cpu_alloc = std::chrono::high_resolution_clock::now();
    Matrix h_A, h_B, h_C, h_C_cpu;

    // Set matrix dimensions
    h_A.width = A_WIDTH; h_A.height = A_HEIGHT;
    h_B.width = B_WIDTH; h_B.height = B_HEIGHT;
    h_C.width = C_WIDTH; h_C.height = C_HEIGHT;
    h_C_cpu.width = C_WIDTH; h_C_cpu.height = C_HEIGHT;

    // Allocate memory for the elements
    int size_A = h_A.width * h_A.height * sizeof(float);
    int size_B = h_B.width * h_B.height * sizeof(float);
    int size_C = h_C.width * h_C.height * sizeof(float);

    h_A.elements = (float*)malloc(size_A);
    h_B.elements = (float*)malloc(size_B);
    h_C.elements = (float*)malloc(size_C);
    h_C_cpu.elements = (float*)malloc(size_C);

    // Initialize host matrices A and B with random values
    for (int i = 0; i < h_A.width * h_A.height; ++i) {
        h_A.elements[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < h_B.width * h_B.height; ++i) {
        h_B.elements[i] = (float)rand() / RAND_MAX;
    }

    auto stop_cpu_alloc = std::chrono::high_resolution_clock::now();
    auto duration_cpu_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu_alloc - start_cpu_alloc);
    std::cout << "-> Host allocation time: " << duration_cpu_alloc.count() << " microseconds\n" << std::endl;

    // --- 3. Allocate Device (GPU) Memory ---
    std::cout << "Allocating device memory..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start));
    Matrix d_A, d_B, d_C;

    // Copy dimensions to device matrix structs
    d_A.width = h_A.width; d_A.height = h_A.height;
    d_B.width = h_B.width; d_B.height = h_B.height;
    d_C.width = h_C.width; d_C.height = h_C.height;

    // Allocate memory on the GPU for the elements
    CUDA_CHECK(cudaMalloc(&d_A.elements, size_A));
    CUDA_CHECK(cudaMalloc(&d_B.elements, size_B));
    CUDA_CHECK(cudaMalloc(&d_C.elements, size_C));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "-> Device allocation time: " << milliseconds << " ms\n" << std::endl;


    // --- 4. Copy Data from Host to Device ---
    std::cout << "Copying data from host to device..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(d_A.elements, h_A.elements, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B.elements, h_B.elements, size_B, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "-> Host to Device copy time: " << milliseconds << " ms\n" << std::endl;

    // --- 5. Launch the CUDA Kernel ---
    std::cout << "Launching CUDA kernel..." << std::endl;

    // Define the thread block dimensions
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    // Define the grid dimensions
    // The grid should be large enough to cover the entire output matrix C
    dim3 dimGrid(ceil((float)C_WIDTH / dimBlock.x), ceil((float)C_HEIGHT / dimBlock.y));

    CUDA_CHECK(cudaEventRecord(start));

    // Launch the kernel on the device
    MatMulKernel << <dimGrid, dimBlock >> > (d_A, d_B, d_C);

    // Check for any errors during kernel launch
    CUDA_CHECK(cudaGetLastError());

    // Synchronize the device to ensure the kernel has finished execution
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Kernel execution finished." << std::endl;
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "-> Kernel execution time: " << milliseconds << " ms\n" << std::endl;


    // --- 6. Copy Results from Device to Host ---
    std::cout << "Copying results from device to host..." << std::endl;
    CUDA_CHECK(cudaMemcpy(h_C.elements, d_C.elements, size_C, cudaMemcpyDeviceToHost));

    // --- 7. Verification ---
    std::cout << "Verifying results against CPU computation..." << std::endl;
    auto start_cpu_verify = std::chrono::high_resolution_clock::now();
    MatMulCPU(h_A, h_B, h_C_cpu);

    auto stop_cpu_verify = std::chrono::high_resolution_clock::now();
    auto duration_cpu_verify = std::chrono::duration_cast<std::chrono::milliseconds>(stop_cpu_verify - start_cpu_verify);
    std::cout << "-> CPU verification time: " << duration_cpu_verify.count() << " milliseconds\n" << std::endl;


    // Compare the GPU result with the CPU result
    bool success = true;
    for (int i = 0; i < C_WIDTH * C_HEIGHT; ++i) {
        if (fabs(h_C.elements[i] - h_C_cpu.elements[i]) > 1e-4) {
            std::cerr << "Verification FAILED at index " << i << "!" << std::endl;
            std::cerr << "GPU result: " << h_C.elements[i] << ", CPU result: " << h_C_cpu.elements[i] << std::endl;
            success = false;
            break;
        }
    }
    if (success) {
        std::cout << "Verification PASSED!" << std::endl;
    }

    // --- 8. Free Memory ---
    std::cout << "Freeing memory..." << std::endl;
    // Free device memory
    CUDA_CHECK(cudaFree(d_A.elements));
    CUDA_CHECK(cudaFree(d_B.elements));
    CUDA_CHECK(cudaFree(d_C.elements));

    // Free host memory
    free(h_A.elements);
    free(h_B.elements);
    free(h_C.elements);
    free(h_C_cpu.elements);

    return 0;
}
