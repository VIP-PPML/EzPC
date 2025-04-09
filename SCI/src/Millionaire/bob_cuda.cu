#include <cuda_runtime.h>
#include <cstdint>

__global__ void pack_ot_selection_kernel(const uint8_t* a, const uint8_t* b, uint8_t* ot_selection, int num_triples) {
  // Correspond to "i/2" in CPU loop: each thread processes one ot_selection[i/2]
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // i = 2 * idx
  int i = idx * 2;

  // Bounds check, same as CPU loop's end condition
  if (i + 1 < num_triples) {
    // Same bit-packing as inside CPU loop
    ot_selection[idx] = (b[i + 1] << 3) | (a[i + 1] << 2) | (b[i] << 1) | a[i];
  }
}

extern "C" void compute_bob_ot_gpu(const uint8_t* a_host, const uint8_t* b_host, uint8_t* ot_selection_host, int num_triples) {

  // Allocate GPU memory
  int half = num_triples / 2;
  uint8_t *a_dev, *b_dev, *ot_sel_dev;
  cudaMalloc(&a_dev, num_triples * sizeof(uint8_t));
  cudaMalloc(&b_dev, num_triples * sizeof(uint8_t));
  cudaMalloc(&ot_sel_dev, half * sizeof(uint8_t));

  // Copy input from host to device (a, b)
  cudaMemcpy(a_dev, a_host, num_triples * sizeof(uint8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b_host, num_triples * sizeof(uint8_t), cudaMemcpyHostToDevice);

  // Kernel launch configuration
  int threads_per_block = 256;
  int blocks = (half + threads_per_block - 1) / threads_per_block;

  // Launch CUDA kernel to compute packed ot_selection
  pack_ot_selection_kernel<<<blocks, threads_per_block>>>(a_dev, b_dev, ot_sel_dev, num_triples);
  cudaDeviceSynchronize(); // Ensure kernel is done

  // Copy result from device back to host 
  cudaMemcpy(ot_selection_host, ot_sel_dev, half * sizeof(uint8_t), cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(ot_sel_dev);
}

