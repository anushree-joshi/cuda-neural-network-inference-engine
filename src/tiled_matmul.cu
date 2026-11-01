
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_SIZE 16


// ============================================================
// CPU MATRIX MULTIPLICATION
// ============================================================

void matmulCPU(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {

            float sum = 0.0f;

            for (int k = 0; k < K; k++) {
                sum += A[row * K + k]
                     * B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}


// ============================================================
// TILED CUDA MATRIX MULTIPLICATION
// ============================================================

__global__ void matmulTiled(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
) {

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];


    int row =
        blockIdx.y * TILE_SIZE
        + threadIdx.y;

    int col =
        blockIdx.x * TILE_SIZE
        + threadIdx.x;


    float sum = 0.0f;


    // Number of tiles needed across K dimension
    int numTiles =
        (K + TILE_SIZE - 1) / TILE_SIZE;


    for (int tile = 0; tile < numTiles; tile++) {

        // ----------------------------------------------------
        // LOAD TILE OF A INTO SHARED MEMORY
        // ----------------------------------------------------

        int aCol =
            tile * TILE_SIZE
            + threadIdx.x;

        if (row < M && aCol < K) {

            tileA[threadIdx.y][threadIdx.x] =
                A[row * K + aCol];

        } else {

            tileA[threadIdx.y][threadIdx.x] =
                0.0f;
        }


        // ----------------------------------------------------
        // LOAD TILE OF B INTO SHARED MEMORY
        // ----------------------------------------------------

        int bRow =
            tile * TILE_SIZE
            + threadIdx.y;

        if (bRow < K && col < N) {

            tileB[threadIdx.y][threadIdx.x] =
                B[bRow * N + col];

        } else {

            tileB[threadIdx.y][threadIdx.x] =
                0.0f;
        }


        // Wait until ALL threads have loaded their values

        __syncthreads();


        // ----------------------------------------------------
        // COMPUTE USING SHARED MEMORY
        // ----------------------------------------------------

        for (int k = 0; k < TILE_SIZE; k++) {

            sum +=
                tileA[threadIdx.y][k]
                *
                tileB[k][threadIdx.x];
        }


        // Don't overwrite tiles while another thread uses them

        __syncthreads();
    }


    // --------------------------------------------------------
    // WRITE RESULT
    // --------------------------------------------------------

    if (row < M && col < N) {

        C[row * N + col] = sum;
    }
}


// ============================================================
// MAIN
// ============================================================

int main() {

    const int M = 32;
    const int K = 32;
    const int N = 32;

    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);


    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_cpuC = new float[M * N];
    float* h_gpuC = new float[M * N];


    // Fill matrices with predictable values

    for (int i = 0; i < M * K; i++) {
        h_A[i] = static_cast<float>((i % 5) + 1);
    }

    for (int i = 0; i < K * N; i++) {
        h_B[i] = static_cast<float>((i % 7) + 1);
    }


    // ========================================================
    // CPU
    // ========================================================

    matmulCPU(
        h_A,
        h_B,
        h_cpuC,
        M,
        K,
        N
    );


    // ========================================================
    // GPU MEMORY
    // ========================================================

    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);


    cudaMemcpy(
        d_A,
        h_A,
        bytesA,
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_B,
        h_B,
        bytesB,
        cudaMemcpyHostToDevice
    );


    // ========================================================
    // CUDA CONFIGURATION
    // ========================================================

    dim3 threads(
        TILE_SIZE,
        TILE_SIZE
    );

    dim3 blocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );


    // ========================================================
    // KERNEL
    // ========================================================

    matmulTiled<<<blocks, threads>>>(
        d_A,
        d_B,
        d_C,
        M,
        K,
        N
    );

    cudaDeviceSynchronize();


    cudaMemcpy(
        h_gpuC,
        d_C,
        bytesC,
        cudaMemcpyDeviceToHost
    );


    // ========================================================
    // VERIFY
    // ========================================================

    bool correct = true;
    float maxDifference = 0.0f;

    for (int i = 0; i < M * N; i++) {

        float difference =
            std::fabs(h_cpuC[i] - h_gpuC[i]);

        if (difference > maxDifference) {
            maxDifference = difference;
        }

        if (difference > 1e-4) {
            correct = false;
        }
    }


    if (correct) {

        std::cout
            << "CPU and tiled GPU results match!"
            << std::endl;

    } else {

        std::cout
            << "ERROR: results do not match."
            << std::endl;
    }


    std::cout
        << "Maximum difference: "
        << maxDifference
        << std::endl;


    // ========================================================
    // CLEANUP
    // ========================================================

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_cpuC;
    delete[] h_gpuC;

    return 0;
}
