
//Cuda_Aula_03.cu

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


// Kernel Add !!
__global__ void addThread(int* a, int* b, int* c) {
	int idx = threadIdx.x;
	c[idx] = a[idx] + b[idx];
	printf("Hello from CUDA Block X %d\n", blockIdx.x);
	printf("Hello from CUDA thread X %d\n", threadIdx.x);

}

// Kenel with Blocks
__global__ void add_block(int* a, int* b, int* c) {
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
	printf("Hello from CUDA Block X %d\n", blockIdx.x);
	printf("Hello from CUDA thread X %d\n", threadIdx.x);
}


__global__ void add_Index(int* a, int* b, int* c) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	c[index] = a[index] + b[index];
	printf("Hello from CUDA Index X %d\n", index);
	printf("Hello from CUDA Block X %d\n", blockIdx.x);
	printf("Hello from CUDA thread X %d\n", threadIdx.x);
}

// Kernel Combined Block and Threads
// n= 7
//grid =8
__global__ void add_Index_Size(int* a, int* b, int* c, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < n){
		c[index] = a[index] + b[index];
		}
	printf("Hello from CUDA Index X %d\n", index);
	printf("Hello from CUDA Block X %d\n", blockIdx.x);
	printf("Hello from CUDA thread X %d\n", threadIdx.x);
}


// CUDA Flow Steps 2025
//#define N 10             // Parallel Problem !!

// With More Parallel computing power, Solve bigger Problems!! 

#define N (7) // Bigger Problem
//#define THREADS_PER_BLOCK 4
#define M 4



int main(int argc, char** argv) {

	int size = N * sizeof(int);

	// Step 01: Allocate memory space in host(CPU) for data
	int h_a[N], h_b[N], h_c[N]; // Host 
	//malloc 

	// Step 02 : Allocate memory space in device(GPU) for data
	int* d_a, * d_b, * d_c; // Device

	// Allocate device memory
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	// Initialize host arrays
	for (int i = 0; i < N; ++i) {
		h_a[i] = i;
		h_b[i] = i * 2;
	}

	// Step 03 : Copy data to GPU
	// Copy data from host to device
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);


	// Step 04 : Call “kernel” routine to execute on GPU(with CUDA syntax that defines no of threads and their physical structure)
	std::cout << "Hello Wold From CPU !\n";

	// Launch kernel
	
	//addThread << <1, N >> > (d_a, d_b, d_c);
	//add_block << <N, 1 >> > (d_a, d_b, d_c);


	//add_Index << <N / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_a, d_b, d_c);

	/*N = Problem!!!
	THREADS_PER_BLOCK= M = 4 

		N/M - 7/4 =1 = 4 threads - 3
		N+M-1 / M
		7 +4-1 / 7
		7+ 3 /7
		10/7 = 2 - 4*3 =8 - 7*/

	int blocks = N + (M - 1) / M;
	
	add_Index_Size << < blocks, M >> > (d_a, d_b, d_c, N);
	

	printf("Hello from GPU (CPU Main)!\n");
	// Step 05 : Transfer results from GPU to CPU

	// Copy data from device to host
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);


	// Show Results
	// Print results
	for (int i = 0; i < N; ++i) {
		std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
	}

	// Step 06 : Free memory space in device(GPU)
	// Free device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// Step 07 : Free memory space in host(CPU)

	return 0;
}

