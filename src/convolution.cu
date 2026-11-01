
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define KERNEL_SIZE 3

// ---------------- CPU CONVOLUTION ----------------

void convCPU(
    const float* input,
    const float* filter,
    float* output,
    int H,
    int W
) {
    int outH = H - KERNEL_SIZE + 1;
    int outW = W - KERNEL_SIZE + 1;

    for (int y = 0; y < outH; y++) {
        for (int x = 0; x < outW; x++) {

            float sum = 0.0f;

            for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                for (int kx = 0; kx < KERNEL_SIZE; kx++) {

                    sum +=
                        input[(y + ky) * W + (x + kx)]
                        *
                        filter[ky * KERNEL_SIZE + kx];
                }
            }

            output[y * outW + x] = sum;
        }
    }
}


// ---------------- GPU CONVOLUTION ----------------

__global__ void convGPU(
    const float* input,
    const float* filter,
    float* output,
    int H,
    int W
) {
    int x =
        blockIdx.x * blockDim.x + threadIdx.x;

    int y =
        blockIdx.y * blockDim.y + threadIdx.y;

    int outH = H - KERNEL_SIZE + 1;
    int outW = W - KERNEL_SIZE + 1;

    if (x < outW && y < outH) {

        float sum = 0.0f;

        for (int ky = 0; ky < KERNEL_SIZE; ky++) {
            for (int kx = 0; kx < KERNEL_SIZE; kx++) {

                sum +=
                    input[(y + ky) * W + (x + kx)]
                    *
                    filter[ky * KERNEL_SIZE + kx];
            }
        }

        output[y * outW + x] = sum;
    }
}


// ---------------- MAIN ----------------

int main() {

    const int H = 5;
    const int W = 5;

    const int outH = H - KERNEL_SIZE + 1;
    const int outW = W - KERNEL_SIZE + 1;

    float h_input[H * W] = {
        1,2,3,4,5,
        6,7,8,9,10,
        11,12,13,14,15,
        16,17,18,19,20,
        21,22,23,24,25
    };

    // Simple edge-detection style filter
    float h_filter[9] = {
        1, 0, -1,
        1, 0, -1,
        1, 0, -1
    };

    float h_cpu[outH * outW];
    float h_gpu[outH * outW];

    // CPU
    convCPU(h_input, h_filter, h_cpu, H, W);

    // GPU memory
    float *d_input, *d_filter, *d_output;

    cudaMalloc(&d_input, H * W * sizeof(float));
    cudaMalloc(&d_filter, 9 * sizeof(float));
    cudaMalloc(&d_output, outH * outW * sizeof(float));

    cudaMemcpy(
        d_input,
        h_input,
        H * W * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_filter,
        h_filter,
        9 * sizeof(float),
        cudaMemcpyHostToDevice
    );

    // 2D CUDA grid
    dim3 threads(16,16);

    dim3 blocks(
        (outW + 15) / 16,
        (outH + 15) / 16
    );

    convGPU<<<blocks,threads>>>(
        d_input,
        d_filter,
        d_output,
        H,
        W
    );

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_gpu,
        d_output,
        outH * outW * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    // Verify
    bool correct = true;

    for (int i = 0; i < outH * outW; i++) {

        if (fabs(h_cpu[i] - h_gpu[i]) > 1e-5)
            correct = false;
    }

    std::cout << "CPU output:\n";

    for (int y = 0; y < outH; y++) {
        for (int x = 0; x < outW; x++)
            std::cout << h_cpu[y*outW+x] << " ";

        std::cout << "\n";
    }

    std::cout << "\nGPU output:\n";

    for (int y = 0; y < outH; y++) {
        for (int x = 0; x < outW; x++)
            std::cout << h_gpu[y*outW+x] << " ";

        std::cout << "\n";
    }

    std::cout <<
        (correct
        ? "\nCPU and GPU convolution match!\n"
        : "\nERROR!\n");

    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    return 0;
}
