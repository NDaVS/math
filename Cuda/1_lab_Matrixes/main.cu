#include <stdio.h>
#include <stdio.h>
#include <time.h>

#define BLOCK_SIZE 16
#define N 10

__global__ void matMult (float *a, float *b, int n, float *c){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;
    int ia = n * BLOCK_SIZE * by + n * ty;
    int ib = BLOCK_SIZE * bx + tx;

    for (int k = 0; k < n; k++){
        sum += a[ia + k] * b[k * n + ib];
    }

    int ic = n * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    c[ic + n * ty + tx] = sum;
}
void printMatrix(float *matrix, int n, int numElements) {
    for (int i = 0; i < numElements && i < n * n; i++) {
        printf("%f ", matrix[i]);
        if ((i + 1) % n == 0) {
            printf("\n"); 
        }
    }
    printf("\n");
}

void matMulCPU(float * a, float * b, float * c, int n){
    for ( int i = 0; i < n; i++){
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++){
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

bool compareMatrices(float * a, float * b, int n){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            if (fabs(a[i * n + j] - b[i * n + j]) > 1e-3){
                return false;
            }
        }
    }
    return true;
}

int main (int argc, char * argv []){
    int numBytes = N * N * sizeof(float);

    float * a = new float [numBytes];
    float * b = new float [numBytes];
    float * c_gpu = new float [numBytes];
    float * c_cpu = new float [numBytes];

    for (int i = 0; i < N; i ++){
        for (int j = 0; j < N; j++){
            int k=  N *i + j;
            a[k] = static_cast<float>(rand()) / RAND_MAX;
            b[k] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    float *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, numBytes);
    cudaMalloc((void**)&d_b, numBytes);
    cudaMalloc((void**)&d_c, numBytes);

    dim3 threads (BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks (N / threads.x, N / threads.y);

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    cudaMemcpy(d_a, a, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, numBytes, cudaMemcpyHostToDevice);


    matMult<<<blocks, threads>>>(d_a, d_b, N, d_c);
    
    cudaMemcpy(c_gpu, d_c, numBytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    printf("GPU Time: %f ms\n", gpuTime);

    clock_t start_cpu = clock();
    matMulCPU(a, b, c_cpu, N);
    clock_t end_cpu = clock();
    double cpuTime = double(end_cpu - start_cpu) / CLOCKS_PER_SEC * 1000;

    printf("CPU Time: %f ms\n", cpuTime);
    if (compareMatrices(c_gpu, c_cpu, N)){
        printf("Matrices are equal\n");
    } else {
        printf("Matrices are not equal\n");
    }

    // printf("Result from GPU:\n");
    // printMatrix(c_gpu, N, 10); 

    // printf("Result from CPU:\n");
    // printMatrix(c_cpu, N, 10); 

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);



    delete a;
    delete b;
    delete c_gpu;
    delete c_cpu;

    return 0;
}