
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

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


// ============================================================
// CUDA ERROR CHECKING
// ============================================================

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error: "                           \
                  << cudaGetErrorString(err)                   \
                  << " at line " << __LINE__ << std::endl;    \
        exit(1);                                               \
    }                                                         \
} while(0)


// ============================================================
// LOAD BINARY FLOAT FILE
// ============================================================

bool loadBinary(
    const char* filename,
    float* data,
    size_t count
) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr
            << "Could not open "
            << filename
            << std::endl;

        return false;
    }

    file.read(
        reinterpret_cast<char*>(data),
        count * sizeof(float)
    );

    return file.good() || file.eof();
}


// ============================================================
// CONVOLUTION
//
// Input:   1 x 28 x 28
// Filters: 8 x 3 x 3
// Output:  8 x 26 x 26
// ============================================================

__global__ void convGPU(
    const float* input,
    const float* filters,
    float* output
) {
    int x =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    int y =
        blockIdx.y * blockDim.y
        + threadIdx.y;

    int f = blockIdx.z;

    if (
        x < CONV_W &&
        y < CONV_H &&
        f < FILTERS
    ) {

        float sum = 0.0f;

        for (int ky = 0; ky < KERNEL; ky++) {

            for (int kx = 0; kx < KERNEL; kx++) {

                float pixel =
                    input[
                        (y + ky) * IMG_W
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
            f * CONV_H * CONV_W
            + y * CONV_W
            + x
        ] = sum;
    }
}


// ============================================================
// ReLU
// ============================================================

__global__ void reluGPU(
    float* data,
    int n
) {
    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (i < n) {
        data[i] = fmaxf(
            0.0f,
            data[i]
        );
    }
}


// ============================================================
// 2x2 MAX POOLING
//
// Input:  8 x 26 x 26
// Output: 8 x 13 x 13
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

    int f = blockIdx.z;

    if (
        x < POOL_W &&
        y < POOL_H &&
        f < FILTERS
    ) {

        int inputX = x * 2;
        int inputY = y * 2;

        float maxValue = -1e30f;

        for (int py = 0; py < 2; py++) {

            for (int px = 0; px < 2; px++) {

                float value =
                    input[
                        f * CONV_H * CONV_W
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
            f * POOL_H * POOL_W
            + y * POOL_W
            + x
        ] = maxValue;
    }
}


// ============================================================
// FULLY CONNECTED LAYER
//
// input:   1352
// weights: 10 x 1352
// output:  10
// ============================================================

__global__ void fullyConnectedGPU(
    const float* input,
    const float* weights,
    float* output
) {
    int neuron =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (neuron < NUM_CLASSES) {

        float sum = 0.0f;

        for (int i = 0; i < FLAT_SIZE; i++) {

            sum +=
                input[i]
                *
                weights[
                    neuron * FLAT_SIZE
                    + i
                ];
        }

        output[neuron] = sum;
    }
}


// ============================================================
// MAIN
// ============================================================

int main() {

    std::cout
        << "CUDA Neural Network Inference Engine\n\n";


    // --------------------------------------------------------
    // HOST DATA
    // --------------------------------------------------------

    std::vector<float> image(
        IMG_H * IMG_W
    );

    std::vector<float> convWeights(
        FILTERS * KERNEL * KERNEL
    );

    std::vector<float> fcWeights(
        NUM_CLASSES * FLAT_SIZE
    );

    std::vector<float> logits(
        NUM_CLASSES
    );


    // --------------------------------------------------------
    // LOAD DATA
    // --------------------------------------------------------

    if (!loadBinary(
            "test_image.bin",
            image.data(),
            image.size()
        )) {
        return 1;
    }

    if (!loadBinary(
            "conv_weights.bin",
            convWeights.data(),
            convWeights.size()
        )) {
        return 1;
    }

    if (!loadBinary(
            "fc_weights.bin",
            fcWeights.data(),
            fcWeights.size()
        )) {
        return 1;
    }


    // --------------------------------------------------------
    // DEVICE POINTERS
    // --------------------------------------------------------

    float* d_image;
    float* d_convWeights;
    float* d_convOutput;
    float* d_poolOutput;
    float* d_fcWeights;
    float* d_logits;


    // --------------------------------------------------------
    // ALLOCATE GPU MEMORY
    // --------------------------------------------------------

    CUDA_CHECK(cudaMalloc(
        &d_image,
        IMG_H * IMG_W * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_convWeights,
        FILTERS * KERNEL * KERNEL
        * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_convOutput,
        FILTERS * CONV_H * CONV_W
        * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_poolOutput,
        FLAT_SIZE * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_fcWeights,
        NUM_CLASSES * FLAT_SIZE
        * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_logits,
        NUM_CLASSES * sizeof(float)
    ));


    // --------------------------------------------------------
    // COPY INPUT + WEIGHTS TO GPU
    // --------------------------------------------------------

    CUDA_CHECK(cudaMemcpy(
        d_image,
        image.data(),
        IMG_H * IMG_W * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_convWeights,
        convWeights.data(),
        convWeights.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_fcWeights,
        fcWeights.data(),
        fcWeights.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));


    // ========================================================
    // 1. CONVOLUTION
    // ========================================================

    dim3 convThreads(16, 16);

    dim3 convBlocks(
        (CONV_W + 15) / 16,
        (CONV_H + 15) / 16,
        FILTERS
    );

    convGPU<<<
        convBlocks,
        convThreads
    >>>(
        d_image,
        d_convWeights,
        d_convOutput
    );

    CUDA_CHECK(cudaGetLastError());


    // ========================================================
    // 2. ReLU
    // ========================================================

    int convElements =
        FILTERS * CONV_H * CONV_W;

    int threads = 256;

    int reluBlocks =
        (convElements + threads - 1)
        / threads;

    reluGPU<<<
        reluBlocks,
        threads
    >>>(
        d_convOutput,
        convElements
    );

    CUDA_CHECK(cudaGetLastError());


    // ========================================================
    // 3. MAX POOL
    // ========================================================

    dim3 poolThreads(16, 16);

    dim3 poolBlocks(
        (POOL_W + 15) / 16,
        (POOL_H + 15) / 16,
        FILTERS
    );

    poolGPU<<<
        poolBlocks,
        poolThreads
    >>>(
        d_convOutput,
        d_poolOutput
    );

    CUDA_CHECK(cudaGetLastError());


    // ========================================================
    // 4. FULLY CONNECTED / MATRIX MULTIPLICATION
    // ========================================================

    fullyConnectedGPU<<<1, 32>>>(
        d_poolOutput,
        d_fcWeights,
        d_logits
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    // --------------------------------------------------------
    // COPY FINAL RESULT BACK
    // --------------------------------------------------------

    CUDA_CHECK(cudaMemcpy(
        logits.data(),
        d_logits,
        NUM_CLASSES * sizeof(float),
        cudaMemcpyDeviceToHost
    ));


    // --------------------------------------------------------
    // ARGMAX
    // --------------------------------------------------------

    int prediction = 0;

    for (
        int i = 1;
        i < NUM_CLASSES;
        i++
    ) {
        if (
            logits[i] >
            logits[prediction]
        ) {
            prediction = i;
        }
    }


    // --------------------------------------------------------
    // PRINT SCORES
    // --------------------------------------------------------

    std::cout
        << "Class scores:\n";

    for (
        int i = 0;
        i < NUM_CLASSES;
        i++
    ) {

        std::cout
            << i
            << ": "
            << logits[i]
            << "\n";
    }


    std::cout
        << "\nPredicted digit: "
        << prediction
        << "\n";


    // --------------------------------------------------------
    // CLEANUP
    // --------------------------------------------------------

    cudaFree(d_image);
    cudaFree(d_convWeights);
    cudaFree(d_convOutput);
    cudaFree(d_poolOutput);
    cudaFree(d_fcWeights);
    cudaFree(d_logits);

    return 0;
}
