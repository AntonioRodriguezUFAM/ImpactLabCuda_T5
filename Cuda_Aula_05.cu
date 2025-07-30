// CUDA Image Blur Filter

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>

#include <chrono> // For timing CPU operations

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Define these before including stb_image.h to act as the implementation
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

// Macro to wrap CUDA API calls and check for errors
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
                  << ": " << cudaGetErrorString(err_) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA Kernel for a Box Blur filter
// Each thread processes one pixel of the output image.
__global__ void blurKernel(const unsigned char* in, unsigned char* out, int width, int height, int channels, int radius) {
    // Calculate the global x and y coordinates of the pixel for this thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check: ensure the thread is within the image dimensions
    if (col < width && row < height) {
        // Accumulators for the color channels (Red, Green, Blue)
        float r_acc = 0.0f, g_acc = 0.0f, b_acc = 0.0f;
        int pixel_count = 0;

        // Iterate over the neighborhood defined by the blur radius
        for (int y = -radius; y <= radius; ++y) {
            for (int x = -radius; x <= radius; ++x) {
                int current_row = row + y;
                int current_col = col + x;

                // Boundary check for the neighborhood pixels
                if (current_row >= 0 && current_row < height && current_col >= 0 && current_col < width) {
                    // Calculate the index of the neighboring pixel in the 1D array
                    int idx = (current_row * width + current_col) * channels;

                    // Accumulate color values
                    r_acc += in[idx + 0];
                    g_acc += in[idx + 1];
                    b_acc += in[idx + 2];
                    // Alpha channel (if present) is ignored for the blur calculation

                    pixel_count++;
                }
            }
        }

        // Calculate the average color value
        int out_idx = (row * width + col) * channels;
        out[out_idx + 0] = static_cast<unsigned char>(r_acc / pixel_count);
        out[out_idx + 1] = static_cast<unsigned char>(g_acc / pixel_count);
        out[out_idx + 2] = static_cast<unsigned char>(b_acc / pixel_count);

        // If there's an alpha channel, copy it directly
        if (channels == 4) {
            out[out_idx + 3] = in[out_idx + 3];
        }
    }
}


int main(int argc, char** argv) {
    // --- 1. Setup and Initialization ---
    const std::string input_filename = "Images/apple.jpg";
    const std::string output_filename = "Images/output.jpg";
    const int blur_radius = 5; // The radius of the blur kernel

    // CUDA events for timing GPU operations
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float milliseconds = 0;

    // --- 2. Load Image and Allocate Host (CPU) Memory ---
    std::cout << "Loading image and allocating host memory..." << std::endl;
    auto start_cpu_alloc = std::chrono::high_resolution_clock::now();

    int width, height, channels;
    // Load image from file. The '0' forces stb_image to decide the number of channels.
    unsigned char* h_img_in = stbi_load(input_filename.c_str(), &width, &height, &channels, 0);
    if (h_img_in == nullptr) {
        std::cerr << "Error: Could not load image '" << input_filename << "'.\n";
        std::cerr << "Please ensure the image exists in the same directory as the executable." << std::endl;
        return 1;
    }
    std::cout << "-> Loaded '" << input_filename << "' (" << width << "x" << height << ", " << channels << " channels)" << std::endl;

    // Allocate host memory for the output image
    size_t img_size = width * height * channels * sizeof(unsigned char);
    unsigned char* h_img_out = (unsigned char*)malloc(img_size);

    auto stop_cpu_alloc = std::chrono::high_resolution_clock::now();
    auto duration_cpu_alloc = std::chrono::duration_cast<std::chrono::microseconds>(stop_cpu_alloc - start_cpu_alloc);
    std::cout << "-> Host allocation and image load time: " << duration_cpu_alloc.count() << " microseconds\n" << std::endl;

    // --- 3. Allocate Device (GPU) Memory ---
    std::cout << "Allocating device memory..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start));

    unsigned char* d_img_in = nullptr, * d_img_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_img_in, img_size));
    CUDA_CHECK(cudaMalloc(&d_img_out, img_size));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "-> Device allocation time: " << milliseconds << " ms\n" << std::endl;

    // --- 4. Copy Data from Host to Device ---
    std::cout << "Copying image from host to device..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(d_img_in, h_img_in, img_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "-> Host to Device copy time: " << milliseconds << " ms\n" << std::endl;

    // --- 5. Launch the CUDA Kernel ---
    std::cout << "Launching CUDA blur kernel..." << std::endl;

    // Define thread block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    CUDA_CHECK(cudaEventRecord(start));

    blurKernel << <gridDim, blockDim >> > (d_img_in, d_img_out, width, height, channels, blur_radius);

    CUDA_CHECK(cudaGetLastError()); // Check for errors during kernel launch

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "-> Kernel execution time: " << milliseconds << " ms\n" << std::endl;

    // --- 6. Copy Results from Device to Host ---
    std::cout << "Copying blurred image from device to host..." << std::endl;
    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemcpy(h_img_out, d_img_out, img_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "-> Device to Host copy time: " << milliseconds << " ms\n" << std::endl;

    // --- 7. Save Image and Free Memory ---
    std::cout << "Saving blurred image..." << std::endl;
    stbi_write_jpg(output_filename.c_str(), width, height, channels, h_img_out, 100); // 100 is quality
    std::cout << "-> Saved '" << output_filename << "'" << std::endl;

    std::cout << "\nFreeing memory..." << std::endl;
    // Free host memory
    stbi_image_free(h_img_in);
    free(h_img_out);

    // Free device memory
    CUDA_CHECK(cudaFree(d_img_in));
    CUDA_CHECK(cudaFree(d_img_out));

    // Destroy CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}