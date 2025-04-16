#include <cuda_runtime.h>
#include <cstdint>

// CUDA KERNEL: Pack (a[i], b[i] into ot_selection[i/2]
// Each thread processes 2 elements from a and b
// Packing layout: ot_selection[i/2] = (b[i+1] << 3) | (a[i+1] << 2) | (b[i] << 1) | a[i];
__global__ void pack_ot_selection_kernel(const uint8_t* a, const uint8_t* b, uint8_t* ot_selection, int num_triples) {
  // Correspond to "i/2" in CPU loop
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // i = 2 * idx
  int i = idx * 2;

  // Bounds check, same as CPU loop's end condition
  if (i + 1 < num_triples) {
    // Same bit-packing as inside CPU loop
    ot_selection[idx] = (b[i + 1] << 3) | (a[i + 1] << 2) | (b[i] << 1) | a[i];
  }
}

// HOST FUNCTION: GPU-based bit-packer for Oblivious Transfer
// Inputs: 
// 	- a_host, b_host: host-side input bit arrays (num_triples bytes each)
//	- ot_selection_host: output array on host, packed bits (num_triples / 2 bytes) 
extern "C" void gpu_pack_ot_selection(const uint8_t* a_host, const uint8_t* b_host, uint8_t* ot_selection_host, int num_triples) {

  int half = num_triples / 2;
  uint8_t *a_dev, *b_dev, *ot_sel_dev;

  // Allocate GPU memory
  cudaMalloc(&a_dev, num_triples * sizeof(uint8_t));
  cudaMalloc(&b_dev, num_triples * sizeof(uint8_t));
  cudaMalloc(&ot_sel_dev, half * sizeof(uint8_t));

  // Copy inputs from host to device (a, b)
  cudaMemcpy(a_dev, a_host, num_triples * sizeof(uint8_t), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b_host, num_triples * sizeof(uint8_t), cudaMemcpyHostToDevice);

  // Kernel launch configuration
  int threads_per_block = 256;
  int blocks = (half + threads_per_block - 1) / threads_per_block;

  // Launch CUDA kernel to compute packed ot_selection
  pack_ot_selection_kernel<<<blocks, threads_per_block>>>(a_dev, b_dev, ot_sel_dev, num_triples);
  cudaDeviceSynchronize(); // Ensure kernel is done

  // Copy packed result from device back to host 
  cudaMemcpy(ot_selection_host, ot_sel_dev, half * sizeof(uint8_t), cudaMemcpyDeviceToHost);

  // Free GPU memory
  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(ot_sel_dev);
}

// CUDA KERNEL: Unpack ot_result[i] into c[2*i] and c[2*i+1]
// Each ot_result byte holds 2 output bits 
// Latout:
//	c[2*i] = ot_result[i] & 1
//	c[2*i+1] = (ot_result[i] >> 1) & 1
__global__ void unpack_ot_result_kernel(const uint8_t* ot_result, uint8_t* c, int num_triples) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i = idx * 2;

  if (i + 1 < num_triples) {
    uint8_t val = ot_result[idx];
    c[i] = val & 1;
    c[i + 1] = (val >> 1) & 1;
  }
}  

// HOST FUNCTION: GPU-based unpacker for OT result
// Inputs:
//	- ot_result_host: packed input bit pairs from OT result (num_triples / 2 bytes)
//	- c_host: output array on host, unpacked bits (num_triples bytes)
extern "C" void gpu_unpack_ot_result(const uint8_t* ot_result_host, uint8_t* c_host, int num_triples) {
  int half = num_triples / 2;
  uint8_t *ot_result_dev, *c_dev;

  // Allocate GPU memory
  cudaMalloc(&ot_result_dev, half * sizeof(uint8_t));
  cudaMalloc(&c_dev, num_triples * sizeof(uint8_t));

  // Copy packed result from host to device
  cudaMemcpy(ot_result_dev, ot_result_host, half * sizeof(uint8_t), cudaMemcpyHostToDevice);

  // Launch kernel to unpack each byte into two bits
  int threads = 256;
  int blocks = (half + threads - 1) / threads;
  unpack_ot_result_kernel<<<blocks, threads>>>(ot_result_dev, c_dev, num_triples);
  cudaDeviceSynchronize();

  // Copy unpacked bits back to host
  cudaMemcpy(c_host, c_dev, num_triples * sizeof(uint8_t), cudaMemcpyDeviceToHost);

  // Free GPU memory
  cudaFree(ot_result_dev);
  cudaFree(c_dev);
}
