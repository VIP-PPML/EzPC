#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>

__global__ void set_leaf_ot_messages(uint8_t *ot_messages, uint8_t digit, int N,
                                     uint8_t mask_cmp, uint8_t mask_eq,
                                     bool greater_than, bool eq = true)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < N; i += stride)
    {
        if (greater_than)
        {
            ot_messages[i] = ((digit > i) ^ mask_cmp);
        }
        else
        {
            ot_messages[i] = ((digit < i) ^ mask_cmp);
        }
        if (eq)
        {
            ot_messages[i] = (ot_messages[i] << 1) | ((digit == i) ^ mask_eq);
        }
    }
}

extern "C" void execute_set_leaf(int num_cmps, uint8_t *ot_messages, uint8_t digit, int N,
                                 uint8_t mask_cmp, uint8_t mask_eq,
                                 bool greater_than, bool eq = true)
{

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

__global__ void set_leaf_ot_messages_two(int num_digits, int num_cmps, int beta_pow, uint8_t *all_leaf_ot_messages, uint8_t *digits, uint8_t *leaf_res_cmp, uint8_t *leaf_res_eq,
                                         bool greater_than)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int X = num_digits;
    int Y = num_cmps;
    int Z = beta_pow;

    int total = X * Y * Z;

    if (index < total)
    {
        int i = index / (Y * Z);
        int j = (index / Z) % Y;
        int k = index % Z;

        // compute start of 'row'
        int base_index = (i * num_cmps + j) * beta_pow;
        int full_index = base_index + k; // actual position

        uint8_t digit = digits[i * num_cmps + j];
        uint8_t mask_cmp = leaf_res_cmp[i * num_cmps + j];

        if (greater_than)
        {
            all_leaf_ot_messages[full_index] = ((digit > k) ^ mask_cmp);
        }
        else
        {
            all_leaf_ot_messages[full_index] = ((digit < k) ^ mask_cmp);
        }

        if (i != 0)
        {
            uint8_t mask_eq = leaf_res_eq[i * num_cmps + j];
            bool eq = true;
            all_leaf_ot_messages[full_index] = (all_leaf_ot_messages[full_index] << 1) | ((digit == k) ^ mask_eq);
        }
    }
}

extern "C" void execute_set_leaf_loops(int num_digits, int num_cmps, int beta_pow, uint8_t **all_leaf_ot_messages, uint8_t *digits, uint8_t *leaf_res_cmp, uint8_t *leaf_res_eq,
                                       bool greater_than)
{

    // Allocate GPU memory for all_leaf_ot_messages (2D array)
    size_t total_size = num_digits * num_cmps * beta_pow * sizeof(uint8_t);
    uint8_t *all_leaf_ot_messages_dev;
    cudaMalloc(&all_leaf_ot_messages_dev, total_size);

    for (int i = 0; i < num_digits * num_cmps; i++)
    {
        // copy data for each leaf_ot_messages[i]
        cudaMemcpy(all_leaf_ot_messages_dev + i * beta_pow, all_leaf_ot_messages[i],
                   beta_pow * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }

    // allocate space and copy digits array to GPU
    uint8_t *dev_digits;
    cudaMalloc(&dev_digits, num_digits * num_cmps * sizeof(uint8_t));
    cudaMemcpy(dev_digits, digits, num_digits * num_cmps * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // allocate space and copy leaf_res_cmp array to GPU
    uint8_t *dev_leaf_res_cmp;
    cudaMalloc(&dev_leaf_res_cmp, num_digits * num_cmps * sizeof(uint8_t));
    cudaMemcpy(dev_leaf_res_cmp, leaf_res_cmp, num_digits * num_cmps * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // allocate space and copy leaf_res_eq array to GPU
    uint8_t *dev_leaf_res_eq;
    cudaMalloc(&dev_leaf_res_eq, num_digits * num_cmps * sizeof(uint8_t));
    cudaMemcpy(dev_leaf_res_eq, leaf_res_eq, num_digits * num_cmps * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // set block size and numBlocks
    int threadsPerBlock = 256;
    int N = num_digits * num_cmps * beta_pow;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // call global kernel function
    set_leaf_ot_messages_two<<<numBlocks, threadsPerBlock>>>(num_digits, num_cmps, beta_pow, all_leaf_ot_messages_dev,
                                                             dev_digits, dev_leaf_res_cmp, dev_leaf_res_eq, greater_than);

    // Copy leaf_ot_messages back to host
    for (int i = 0; i < num_digits * num_cmps; i++)
    {
        cudaMemcpy(all_leaf_ot_messages[i], all_leaf_ot_messages_dev + i * beta_pow,
                   beta_pow * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    }

    // cleanup
    cudaFree(all_leaf_ot_messages_dev);

    cudaFree(dev_digits);
    cudaFree(dev_leaf_res_cmp);
    cudaFree(dev_leaf_res_eq);
}
