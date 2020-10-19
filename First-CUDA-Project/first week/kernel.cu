
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <algorithm>
#include <chrono>
#include <iostream>

class cLimitTimer
{
    std::chrono::time_point<std::chrono::steady_clock> Start;
    double Limit;
public:
    cLimitTimer(double limit) : Limit(limit)
    {
        Start = std::chrono::steady_clock::now();
    }
    ~cLimitTimer()
    {
        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double> diff = end - Start;
        if (diff.count() > Limit)
            std::cout << diff.count() << "s greater than limit " << Limit << "s\n";
        else if(diff.count() > 1.0)
            std::cout << diff.count() << "s\n";
        else
            std::cout << diff.count() * 1000 << "ms\n";

    }
};

cudaError_t addWithCuda(int* B, int* A, unsigned int size);

__global__ void addKernel(int *B, int *A, unsigned int size)
{
    //1;
    for (int i = 0; i < size; i++)
         B[i] = A[i];
}

int main()
{
    auto* timer = new cLimitTimer(1000);
    const int arraySize = 50000000;
    int *A = (int*)malloc(sizeof(int) * arraySize);  
    // chrono here
    std::fill_n(A, arraySize, 1);
    int* B = (int*)malloc(sizeof(int) * arraySize);

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(B, A, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("\nresult = {%d,%d,%d,%d,%d...%d}\n",
    B[0], B[1], B[2], B[3], B[4], B[arraySize-1]);

    free(A);
    free(B);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    printf("total time: ");
    delete timer;
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *B, int *A, unsigned int size)
{
    auto* timer1 = new cLimitTimer(1000);
    int* dev_A = 0;
    int* dev_B = 0;
    cudaError_t cudaStatus;
    cudaEvent_t start, stop; //xxx
    cudaEventCreate(&start); //xxx
    cudaEventCreate(&stop); //xxx
    cudaEvent_t start2, stop2;//xxx
    cudaEventCreate(&start2); //xxx
    cudaEventCreate(&stop2); //xxx
    cudaEvent_t start3, stop3;//xxx
    cudaEventCreate(&start3); //xxx
    cudaEventCreate(&stop3); //xxx
    float milliseconds = 0; //XXX
    float milliseconds2 = 0; //XXX
    float milliseconds3 = 0; //XXX
    float seconds = 0; //XXX
    float seconds2 = 0; //XXX
    float seconds3 = 0; //XXX
    int* B1 = (int*)malloc(sizeof(int) * size);
    printf("initialization time: ");
    delete timer1;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;

    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_A, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    cudaStatus = cudaMalloc((void**)&dev_B, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    timer1 = new cLimitTimer(1000);
    cudaEventRecord(start); //xxx

    cudaStatus = cudaMemcpy(dev_A, A, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy dev_A, A failed!");
        goto Error;
    }
    cudaEventRecord(stop); //xxx 
    printf("Host->GPU... : ");
    delete timer1;

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    timer1 = new cLimitTimer(1000);
    for (int i = 0; i < size; i++)
        B1[i] = A[i];
    printf("processing time on cpu : ");
    delete timer1;

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, 1>>>(dev_B, dev_A, size); // using thread, keep in mind to don't go over max cores.

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaEventRecord(start3); //xxx CUDA stream is simply a sequence of operations that are performed in order on the device
    timer1 = new cLimitTimer(1000);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }
    cudaEventRecord(stop3); //xxx
    printf("cudaDeviceSynchronize: ");
    delete timer1;

    timer1 = new cLimitTimer(1000);
    // Copy output vector from GPU buffer to host memory.
    cudaEventRecord(start2); //xxx
    cudaStatus = cudaMemcpy(B, dev_B, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy B, dev_B failed!");
        goto Error;
    }

    cudaEventRecord(stop2); //xxx
    printf("GPU->Host... : ");
    delete timer1;

    cudaEventSynchronize(stop); //XXX wait that event stop occured (cudaEventRecord) // test if this is needed
    cudaEventSynchronize(stop2); //XXX same for stop2
    cudaEventSynchronize(stop3); //XXX same for stop3
    
    
    printf("\n\nCuda event timers:\n");
    cudaEventElapsedTime(&milliseconds, start, stop); //XXX
    printf("Host->GPU, Ellapsed time: %fms\n", milliseconds); //XXX
    cudaEventElapsedTime(&milliseconds2, start2, stop2); //XXX
    printf("GPU->Host, Ellapsed time: %fms\n", milliseconds2); //XXX
    cudaEventElapsedTime(&milliseconds3, start3, stop3); //XXX
    printf("processing time on GPU, Ellapsed time: %fms\n\n", milliseconds3); //XXX // I think this measure is somehow related to what is happening on the GPU.

Error:
    cudaFree(dev_A);
    cudaFree(dev_B);

    return cudaStatus;
}
