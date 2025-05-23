#include <stdio.h>
#include "curand.h"
#include "curand_kernel.h"

#define BLOCKS 10
#define THREADS 256
#define SEED 12345

__global__ void rng_init(curandState_t *rng_state) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(SEED, index, 0, &rng_state[index]);
}

__global__  void rng_generate(double *vector, curandState *rng_state) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    vector[index] = 0;
    curandState* local_rng  = &rng_state[index];
    for (int i = 0; i < 10; i++) {
        vector[index] += curand_uniform_double(local_rng);
    }

}

__global__ void print_res(double * a) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("vector[%d] = %f\n", index, a[index]);
}

int main() {

    curandState_t *dev_rng_state;
    double *dev_vec;

    cudaMalloc(&dev_rng_state, BLOCKS * THREADS * sizeof(curandState_t));
    cudaMalloc(&dev_vec, BLOCKS * THREADS * sizeof(double));

    rng_init<<<BLOCKS, THREADS>>>(dev_rng_state);

    rng_generate<<<BLOCKS, THREADS>>>(dev_vec, dev_rng_state);
    print_res<<<BLOCKS, THREADS>>>(dev_vec);
    cudaDeviceSynchronize();


    return 0;
}
