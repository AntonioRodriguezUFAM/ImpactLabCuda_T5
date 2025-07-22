
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
using namespace std;

//__global__ void mykernel_01 (){}

__global__ void mykernel(void) {

	//std::cout << "Hello Wold CUDA !\n";
	printf("Hello from GPU (CUDA Kernel)!\n");
	printf("Hello from CUDA thread X %d\n", threadIdx.x);
	printf("Hello from CUDA thread Y %d\n", threadIdx.y);
	printf("Hello from CUDA thread Z %d\n", threadIdx.z);
}




// CUDA Flow Steps 2025

int main(int argc, char** argv) {

	// Step 01: Allocate memory space in host(CPU) for data
	
	// Step 02 : Allocate memory space in device(GPU) for data

	// Step 03 : Copy data to GPU

	// Step 04 : Call “kernel” routine to execute on GPU(with CUDA syntax that defines no of threads and their physical structure)
	std::cout << "Hello Wold From CPU !\n";

	mykernel << <1, 10 >> > ();

		
	printf("Hello from GPU (CPU Main)!\n");
	// Step 05 : Transfer results from GPU to CPU

	// Step 06 : Free memory space in device(GPU)

	// Step 07 : Free memory space in host(CPU)

		return 0;
}