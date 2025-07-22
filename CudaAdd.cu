// Tutorial 7: CUDA Memory Management

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


// Kernel Add !!
__global__ void add(int* a, int* b, int* c) {
	int idx = threadIdx.x;
	c[idx] = a[idx] + b[idx];
}


// CUDA Flow Steps 2025

int main(int argc, char** argv) {
	
	// Vector size
	int N = 10;
	int size = N * sizeof(int);

	// Step 01: Allocate memory space in host(CPU) for data
	int h_a[10], h_b[10], h_c[10]; // Host 

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
	add << <1, N >> > (d_a, d_b, d_c);


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


