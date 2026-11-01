
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

#define BATCH 100
#define IMG_H 28
#define IMG_W 28

#define FILTERS 8
#define KERNEL 3

#define OUT_H 26
#define OUT_W 26


// ============================================================
// LOAD BINARY
// ============================================================

bool loadBinary(
    const char* filename,
    float* data,
    size_t count
) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Could not open "
                  << filename << "\n";
        return false;
    }

    file.read(
        reinterpret_cast<char*>(data),
        count * sizeof(float)
    );

    return true;
}


// ============================================================
// CPU CONVOLUTION
// ============================================================

void convCPU(
    const float* images,
    const float* filters,
    float* output
) {
    for (int b = 0; b < BATCH; b++) {

        for (int f = 0; f < FILTERS; f++) {

            for (int y = 0; y < OUT_H; y++) {

                for (int x = 0; x < OUT_W; x++) {

                    float sum = 0.0f;

                    for (int ky = 0; ky < KERNEL; ky++) {

                        for (int kx = 0; kx < KERNEL; kx++) {

                            float pixel =
                                images[
                                    b * IMG_H * IMG_W
                                    + (y + ky) * IMG_W
                                    + (x + kx)
                                ];

                            float weight =
                                filters[
                                    f * KERNEL * KERNEL
                                    + ky * KERNEL
                                    + kx
                                ];

                            sum += pixel * weight;
                        }
                    }

                    output[
                        b * FILTERS * OUT_H * OUT_W
                        + f * OUT_H * OUT_W
                        + y * OUT_W
                        + x
                    ] = sum;
                }
            }
        }
    }
}


// ============================================================
// GPU CONVOLUTION
// ============================================================

__global__ void convGPU(
    const float* images,
    const float* filters,
    float* output
) {
    int x =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    int y =
        blockIdx.y * blockDim.y
        + threadIdx.y;

    int z = blockIdx.z;

    int b = z / FILTERS;
    int f = z % FILTERS;

    if (
        b < BATCH &&
        f < FILTERS &&
        x < OUT_W &&
        y < OUT_H
    ) {

        float sum = 0.0f;

        for (int ky = 0; ky < KERNEL; ky++) {

            for (int kx = 0; kx < KERNEL; kx++) {

                float pixel =
                    images[
                        b * IMG_H * IMG_W
                        + (y + ky) * IMG_W
                        + (x + kx)
                    ];

                float weight =
                    filters[
                        f * KERNEL * KERNEL
                        + ky * KERNEL
                        + kx
                    ];

                sum += pixel * weight;
            }
        }

        output[
            b * FILTERS * OUT_H * OUT_W
            + f * OUT_H * OUT_W
            + y * OUT_W
            + x
        ] = sum;
    }
}


// ============================================================
// MAIN
// ============================================================

int main() {

    const size_t imageCount =
        BATCH * IMG_H * IMG_W;

    const size_t filterCount =
        FILTERS * KERNEL * KERNEL;

    const size_t outputCount =
        BATCH * FILTERS * OUT_H * OUT_W;


    std::vector<float> images(imageCount);
    std::vector<float> filters(filterCount);

    std::vector<float> cpuOutput(outputCount);
    std::vector<float> gpuOutput(outputCount);


    if (!loadBinary(
        "test_100_images.bin",
        images.data(),
        imageCount
    )) return 1;


    if (!loadBinary(
        "conv_weights.bin",
        filters.data(),
        filterCount
    )) return 1;


    // ========================================================
    // CPU BENCHMARK
    // ========================================================

    auto cpuStart =
        std::chrono::high_resolution_clock::now();

    convCPU(
        images.data(),
        filters.data(),
        cpuOutput.data()
    );

    auto cpuEnd =
        std::chrono::high_resolution_clock::now();


    double cpuMs =
        std::chrono::duration<double, std::milli>(
            cpuEnd - cpuStart
        ).count();


    // ========================================================
    // GPU MEMORY
    // ========================================================

    float* d_images;
    float* d_filters;
    float* d_output;

    cudaMalloc(
        &d_images,
        imageCount * sizeof(float)
    );

    cudaMalloc(
        &d_filters,
        filterCount * sizeof(float)
    );

    cudaMalloc(
        &d_output,
        outputCount * sizeof(float)
    );


    cudaMemcpy(
        d_images,
        images.data(),
        imageCount * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_filters,
        filters.data(),
        filterCount * sizeof(float),
        cudaMemcpyHostToDevice
    );


    dim3 threads(16, 16);

    dim3 blocks(
        (OUT_W + 15) / 16,
        (OUT_H + 15) / 16,
        BATCH * FILTERS
    );


    // ========================================================
    // GPU WARM-UP
    // ========================================================

    convGPU<<<blocks, threads>>>(
        d_images,
        d_filters,
        d_output
    );

    cudaDeviceSynchronize();


    // ========================================================
    // GPU BENCHMARK WITH CUDA EVENTS
    // ========================================================

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);

    convGPU<<<blocks, threads>>>(
        d_images,
        d_filters,
        d_output
    );

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);


    float gpuMs = 0.0f;

    cudaEventElapsedTime(
        &gpuMs,
        start,
        stop
    );


    // ========================================================
    // COPY RESULT BACK
    // ========================================================

    cudaMemcpy(
        gpuOutput.data(),
        d_output,
        outputCount * sizeof(float),
        cudaMemcpyDeviceToHost
    );


    // ========================================================
    // CORRECTNESS CHECK
    // ========================================================

    float maxDifference = 0.0f;

    for (size_t i = 0; i < outputCount; i++) {

        float difference =
            std::fabs(
                cpuOutput[i] -
                gpuOutput[i]
            );

        if (difference > maxDifference)
            maxDifference = difference;
    }


    // ========================================================
    // RESULTS
    // ========================================================

    double speedup =
        cpuMs / gpuMs;


    std::cout << "\n===== BENCHMARK =====\n";

    std::cout
        << "Images: "
        << BATCH
        << "\n";

    std::cout
        << "CPU convolution: "
        << cpuMs
        << " ms\n";

    std::cout
        << "GPU convolution: "
        << gpuMs
        << " ms\n";

    std::cout
        << "Speedup: "
        << speedup
        << "x\n";

    std::cout
        << "Maximum difference: "
        << maxDifference
        << "\n";


    // ========================================================
    // CLEANUP
    // ========================================================

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_images);
    cudaFree(d_filters);
    cudaFree(d_output);

    return 0;
}
