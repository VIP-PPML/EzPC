#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>

__global__
void set_leaf_ot_messages(uint8_t *ot_messages, uint8_t digit, int N,
                          uint8_t mask_cmp, uint8_t mask_eq,
                          bool greater_than, bool eq = true) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i+= stride) {
        if (greater_than) {
            ot_messages[i] = ((digit > i) ^ mask_cmp);
        } else {
            ot_messages[i] = ((digit < i) ^ mask_cmp);
        }
        if (eq) {
            ot_messages[i] = (ot_messages[i] << 1) | ((digit == i) ^ mask_eq);
        }
    }
}

extern "C" void execute_set_leaf(int num_cmps, uint8_t *ot_messages, uint8_t digit, int N,
                                 uint8_t mask_cmp, uint8_t mask_eq,
                                 bool greater_than, bool eq = true) {

    // Allocate GPU memory
    uint8_t *ot_messages_dev;
    cudaMalloc(&ot_messages_dev, N * sizeof(uint8_t));
    cudaMemcpy(ot_messages_dev, ot_messages, N * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // set block size and numBlocks
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // call global kernel function
    set_leaf_ot_messages<<<numBlocks, blockSize>>>(ot_messages_dev,
                           digit, N, mask_cmp, mask_eq, greater_than, eq);

    // copy results on device back to host
    cudaMemcpy(ot_messages, ot_messages_dev, N * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(ot_messages_dev);
}
