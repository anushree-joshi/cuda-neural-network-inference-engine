
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

#define CONV_H 26
#define CONV_W 26

#define POOL_H 13
#define POOL_W 13

#define FLAT_SIZE (FILTERS * POOL_H * POOL_W)
#define NUM_CLASSES 10


#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error: "                           \
                  << cudaGetErrorString(err)                   \
                  << " line " << __LINE__ << "\n";             \
        exit(1);                                               \
    }                                                         \
} while(0)


// ============================================================
// LOAD FLOAT BINARY FILE
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

            for (int y = 0; y < CONV_H; y++) {

                for (int x = 0; x < CONV_W; x++) {

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
                        b * FILTERS * CONV_H * CONV_W
                        + f * CONV_H * CONV_W
                        + y * CONV_W
                        + x
                    ] = sum;
                }
            }
        }
    }
}


// ============================================================
// CPU ReLU
// ============================================================

void reluCPU(
    float* data,
    size_t count
) {
    for (size_t i = 0; i < count; i++) {
        data[i] = std::max(
            0.0f,
            data[i]
        );
    }
}


// ============================================================
// CPU POOLING
// ============================================================

void poolCPU(
    const float* input,
    float* output
) {
    for (int b = 0; b < BATCH; b++) {

        for (int f = 0; f < FILTERS; f++) {

            for (int y = 0; y < POOL_H; y++) {

                for (int x = 0; x < POOL_W; x++) {

                    float maxValue = -1e30f;

                    int inputY = y * 2;
                    int inputX = x * 2;

                    for (int py = 0; py < 2; py++) {

                        for (int px = 0; px < 2; px++) {

                            float value =
                                input[
                                    b * FILTERS * CONV_H * CONV_W
                                    + f * CONV_H * CONV_W
                                    + (inputY + py) * CONV_W
                                    + (inputX + px)
                                ];

                            maxValue =
                                std::max(
                                    maxValue,
                                    value
                                );
                        }
                    }

                    output[
                        b * FLAT_SIZE
                        + f * POOL_H * POOL_W
                        + y * POOL_W
                        + x
                    ] = maxValue;
                }
            }
        }
    }
}


// ============================================================
// CPU FULLY CONNECTED
// ============================================================

void fullyConnectedCPU(
    const float* input,
    const float* weights,
    float* output
) {
    for (int b = 0; b < BATCH; b++) {

        for (int neuron = 0;
             neuron < NUM_CLASSES;
             neuron++) {

            float sum = 0.0f;

            for (int i = 0;
                 i < FLAT_SIZE;
                 i++) {

                sum +=
                    input[
                        b * FLAT_SIZE + i
                    ]
                    *
                    weights[
                        neuron * FLAT_SIZE + i
                    ];
            }

            output[
                b * NUM_CLASSES
                + neuron
            ] = sum;
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
        x < CONV_W &&
        y < CONV_H
    ) {

        float sum = 0.0f;

        for (int ky = 0; ky < KERNEL; ky++) {

            for (int kx = 0; kx < KERNEL; kx++) {

                sum +=
                    images[
                        b * IMG_H * IMG_W
                        + (y + ky) * IMG_W
                        + (x + kx)
                    ]
                    *
                    filters[
                        f * KERNEL * KERNEL
                        + ky * KERNEL
                        + kx
                    ];
            }
        }

        output[
            b * FILTERS * CONV_H * CONV_W
            + f * CONV_H * CONV_W
            + y * CONV_W
            + x
        ] = sum;
    }
}


// ============================================================
// GPU ReLU
// ============================================================

__global__ void reluGPU(
    float* data,
    int count
) {
    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (i < count) {
        data[i] =
            fmaxf(
                0.0f,
                data[i]
            );
    }
}


// ============================================================
// GPU POOLING
// ============================================================

__global__ void poolGPU(
    const float* input,
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
        x < POOL_W &&
        y < POOL_H
    ) {

        int inputX = x * 2;
        int inputY = y * 2;

        float maxValue = -1e30f;

        for (int py = 0; py < 2; py++) {

            for (int px = 0; px < 2; px++) {

                float value =
                    input[
                        b * FILTERS * CONV_H * CONV_W
                        + f * CONV_H * CONV_W
                        + (inputY + py) * CONV_W
                        + (inputX + px)
                    ];

                maxValue =
                    fmaxf(
                        maxValue,
                        value
                    );
            }
        }

        output[
            b * FLAT_SIZE
            + f * POOL_H * POOL_W
            + y * POOL_W
            + x
        ] = maxValue;
    }
}


// ============================================================
// GPU FULLY CONNECTED
// ============================================================

__global__ void fullyConnectedGPU(
    const float* input,
    const float* weights,
    float* output
) {
    int neuron =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    int b = blockIdx.y;

    if (
        b < BATCH &&
        neuron < NUM_CLASSES
    ) {

        float sum = 0.0f;

        for (int i = 0;
             i < FLAT_SIZE;
             i++) {

            sum +=
                input[
                    b * FLAT_SIZE + i
                ]
                *
                weights[
                    neuron * FLAT_SIZE + i
                ];
        }

        output[
            b * NUM_CLASSES
            + neuron
        ] = sum;
    }
}


// ============================================================
// MAIN
// ============================================================

int main() {

    const size_t imageCount =
        BATCH * IMG_H * IMG_W;

    const size_t convWeightCount =
        FILTERS * KERNEL * KERNEL;

    const size_t convCount =
        BATCH * FILTERS * CONV_H * CONV_W;

    const size_t poolCount =
        BATCH * FLAT_SIZE;

    const size_t fcWeightCount =
        NUM_CLASSES * FLAT_SIZE;

    const size_t logitCount =
        BATCH * NUM_CLASSES;


    std::vector<float> images(imageCount);
    std::vector<float> convWeights(convWeightCount);
    std::vector<float> fcWeights(fcWeightCount);

    std::vector<float> cpuConv(convCount);
    std::vector<float> cpuPool(poolCount);
    std::vector<float> cpuLogits(logitCount);

    std::vector<float> gpuLogits(logitCount);


    if (!loadBinary(
        "test_100_images.bin",
        images.data(),
        imageCount
    )) return 1;

    if (!loadBinary(
        "conv_weights.bin",
        convWeights.data(),
        convWeightCount
    )) return 1;

    if (!loadBinary(
        "fc_weights.bin",
        fcWeights.data(),
        fcWeightCount
    )) return 1;


    // ========================================================
    // CPU FULL INFERENCE
    // ========================================================

    auto cpuStart =
        std::chrono::high_resolution_clock::now();


    convCPU(
        images.data(),
        convWeights.data(),
        cpuConv.data()
    );

    reluCPU(
        cpuConv.data(),
        convCount
    );

    poolCPU(
        cpuConv.data(),
        cpuPool.data()
    );

    fullyConnectedCPU(
        cpuPool.data(),
        fcWeights.data(),
        cpuLogits.data()
    );


    auto cpuEnd =
        std::chrono::high_resolution_clock::now();


    double cpuMs =
        std::chrono::duration<double, std::milli>(
            cpuEnd - cpuStart
        ).count();


    // ========================================================
    // GPU ALLOCATION
    // ========================================================

    float* d_images;
    float* d_convWeights;
    float* d_conv;
    float* d_pool;
    float* d_fcWeights;
    float* d_logits;


    CUDA_CHECK(cudaMalloc(
        &d_images,
        imageCount * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_convWeights,
        convWeightCount * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_conv,
        convCount * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_pool,
        poolCount * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_fcWeights,
        fcWeightCount * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_logits,
        logitCount * sizeof(float)
    ));


    CUDA_CHECK(cudaMemcpy(
        d_images,
        images.data(),
        imageCount * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_convWeights,
        convWeights.data(),
        convWeightCount * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_fcWeights,
        fcWeights.data(),
        fcWeightCount * sizeof(float),
        cudaMemcpyHostToDevice
    ));


    // ========================================================
    // GPU CONFIGURATION
    // ========================================================

    dim3 convThreads(16,16);

    dim3 convBlocks(
        (CONV_W + 15) / 16,
        (CONV_H + 15) / 16,
        BATCH * FILTERS
    );


    int threads = 256;

    int reluBlocks =
        (convCount + threads - 1)
        / threads;


    dim3 poolThreads(16,16);

    dim3 poolBlocks(
        (POOL_W + 15) / 16,
        (POOL_H + 15) / 16,
        BATCH * FILTERS
    );


    dim3 fcThreads(32);
    dim3 fcBlocks(1, BATCH);


    // ========================================================
    // GPU WARM-UP
    // ========================================================

    convGPU<<<convBlocks,convThreads>>>(
        d_images,
        d_convWeights,
        d_conv
    );

    reluGPU<<<reluBlocks,threads>>>(
        d_conv,
        convCount
    );

    poolGPU<<<poolBlocks,poolThreads>>>(
        d_conv,
        d_pool
    );

    fullyConnectedGPU<<<fcBlocks,fcThreads>>>(
        d_pool,
        d_fcWeights,
        d_logits
    );

    CUDA_CHECK(cudaDeviceSynchronize());


    // ========================================================
    // GPU FULL INFERENCE TIMING
    // ========================================================

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);


    convGPU<<<convBlocks,convThreads>>>(
        d_images,
        d_convWeights,
        d_conv
    );

    reluGPU<<<reluBlocks,threads>>>(
        d_conv,
        convCount
    );

    poolGPU<<<poolBlocks,poolThreads>>>(
        d_conv,
        d_pool
    );

    fullyConnectedGPU<<<fcBlocks,fcThreads>>>(
        d_pool,
        d_fcWeights,
        d_logits
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
    // COPY GPU LOGITS
    // ========================================================

    CUDA_CHECK(cudaMemcpy(
        gpuLogits.data(),
        d_logits,
        logitCount * sizeof(float),
        cudaMemcpyDeviceToHost
    ));


    // ========================================================
    // CORRECTNESS
    // ========================================================

    float maxDifference = 0.0f;

    int matchingPredictions = 0;


    for (int b = 0; b < BATCH; b++) {

        int cpuPrediction = 0;
        int gpuPrediction = 0;

        for (int c = 0;
             c < NUM_CLASSES;
             c++) {

            int index =
                b * NUM_CLASSES + c;

            float difference =
                std::fabs(
                    cpuLogits[index]
                    -
                    gpuLogits[index]
                );

            if (difference > maxDifference)
                maxDifference = difference;


            if (
                cpuLogits[index] >
                cpuLogits[
                    b * NUM_CLASSES
                    + cpuPrediction
                ]
            ) {
                cpuPrediction = c;
            }


            if (
                gpuLogits[index] >
                gpuLogits[
                    b * NUM_CLASSES
                    + gpuPrediction
                ]
            ) {
                gpuPrediction = c;
            }
        }


        if (cpuPrediction == gpuPrediction)
            matchingPredictions++;
    }


    // ========================================================
    // RESULTS
    // ========================================================

    double speedup =
        cpuMs / gpuMs;


    std::cout
        << "\n===== FULL INFERENCE BENCHMARK =====\n";

    std::cout
        << "Images: "
        << BATCH << "\n";

    std::cout
        << "CPU inference: "
        << cpuMs << " ms\n";

    std::cout
        << "GPU inference: "
        << gpuMs << " ms\n";

    std::cout
        << "Speedup: "
        << speedup << "x\n";

    std::cout
        << "CPU/GPU matching predictions: "
        << matchingPredictions
        << "/"
        << BATCH
        << "\n";

    std::cout
        << "Maximum logit difference: "
        << maxDifference
        << "\n";


    // ========================================================
    // CLEANUP
    // ========================================================

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_images);
    cudaFree(d_convWeights);
    cudaFree(d_conv);
    cudaFree(d_pool);
    cudaFree(d_fcWeights);
    cudaFree(d_logits);

    return 0;
}
