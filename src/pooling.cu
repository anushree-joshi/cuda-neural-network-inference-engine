
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define POOL 2

void poolCPU(
    const float* input,
    float* output,
    int H,
    int W
) {
    int outH = H / POOL;
    int outW = W / POOL;

    for (int y = 0; y < outH; y++) {
        for (int x = 0; x < outW; x++) {

            float maxVal = -1e30f;

            for (int py = 0; py < POOL; py++)
                for (int px = 0; px < POOL; px++)
                    maxVal = fmaxf(
                        maxVal,
                        input[
                            (y*POOL+py)*W +
                            (x*POOL+px)
                        ]
                    );

            output[y*outW+x] = maxVal;
        }
    }
}


__global__ void poolGPU(
    const float* input,
    float* output,
    int H,
    int W
) {
    int x =
        blockIdx.x * blockDim.x + threadIdx.x;

    int y =
        blockIdx.y * blockDim.y + threadIdx.y;

    int outH = H / POOL;
    int outW = W / POOL;

    if (x < outW && y < outH) {

        float maxVal = -1e30f;

        for (int py = 0; py < POOL; py++)
            for (int px = 0; px < POOL; px++)
                maxVal = fmaxf(
                    maxVal,
                    input[
                        (y*POOL+py)*W +
                        (x*POOL+px)
                    ]
                );

        output[y*outW+x] = maxVal;
    }
}


int main() {

    const int H = 4;
    const int W = 4;

    float input[16] = {
        1,5,2,4,
        3,8,6,7,
        9,2,4,1,
        5,3,8,6
    };

    float cpu[4];
    float gpu[4];

    poolCPU(input,cpu,H,W);

    float *d_input,*d_output;

    cudaMalloc(&d_input,16*sizeof(float));
    cudaMalloc(&d_output,4*sizeof(float));

    cudaMemcpy(
        d_input,
        input,
        16*sizeof(float),
        cudaMemcpyHostToDevice
    );

    dim3 threads(16,16);
    dim3 blocks(1,1);

    poolGPU<<<blocks,threads>>>(
        d_input,d_output,H,W
    );

    cudaDeviceSynchronize();

    cudaMemcpy(
        gpu,
        d_output,
        4*sizeof(float),
        cudaMemcpyDeviceToHost
    );

    bool correct = true;

    for(int i=0;i<4;i++)
        if(fabs(cpu[i]-gpu[i])>1e-5)
            correct=false;

    std::cout << "Pooling output: ";

    for(int i=0;i<4;i++)
        std::cout << gpu[i] << " ";

    std::cout <<
        (correct
        ? "\nCPU and GPU pooling match!\n"
        : "\nERROR!\n");

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
