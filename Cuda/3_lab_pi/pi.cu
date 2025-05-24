#include <stdio.h>
#include <time.h>
#include "curand.h"
#include "curand_kernel.h"

#define BLOCKS 10
#define THREADS 256

__global__ void rng_init(curandState_t *rng_state) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long seed = clock64() + index;
    curand_init(seed, index, 0, &rng_state[index]);
}

__global__ void rng_generate(double *vector, curandState *rng_state) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    curandState* local_rng = &rng_state[index];
    double x = curand_uniform_double(local_rng);
    double y = curand_uniform_double(local_rng);
    if (x * x + y * y < 1) {
        vector[index] = 1; 
    } else {
        vector[index] = 0; 
    }
}

int main() {
    curandState_t *dev_rng_state;
    int numBytes = BLOCKS * THREADS * sizeof(double);
    double *dev_v;
    double *v = new double[BLOCKS * THREADS]; 
    for (int i = 0; i < BLOCKS * THREADS; i++) {
        v[i] = 0;
    }

    cudaMalloc((void **)&dev_rng_state, BLOCKS * THREADS * sizeof(curandState_t));
    cudaMalloc((void **)&dev_v, numBytes);

    cudaMemcpy(dev_v, v, numBytes, cudaMemcpyHostToDevice);
    
    rng_init<<<BLOCKS, THREADS>>>(dev_rng_state);
    rng_generate<<<BLOCKS, THREADS>>>(dev_v, dev_rng_state);
    
    cudaMemcpy(v, dev_v, numBytes, cudaMemcpyDeviceToHost);

    int count = 0;
    for (int i = 0; i < BLOCKS * THREADS; i++) {
        if (v[i] == 1) {
            count++;
        }
    }

    double pi = 4.0 * count / (BLOCKS * THREADS);
    printf("Estimated value of Pi: %f\n", pi);

    delete[] v;
    cudaFree(dev_v);
    cudaFree(dev_rng_state);
    
    cudaDeviceSynchronize();

    return 0;
}
