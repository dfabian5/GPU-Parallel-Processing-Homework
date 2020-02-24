////////////////////////////////////////////////////////////////////////////////
//
// FILE:        max_parallel_reduct.cu
// DESCRIPTION: uses parallel reduction to find max element in 1000 num array
// AUTHOR:      Dan Fabian
// DATE:        2/23/2020

#include <iostream>
#include <random>
#include <chrono>

using std::cout; using std::endl;
using namespace std::chrono;

// all constants
const int SIZE = 1000;

// kernal func prototypes
__global__ void globalMax(int *vals); // uses global mem
__global__ void interleavingMax(int *vals); // uses interleaving addressing shared mem
__global__ void sequentialMax(int *vals); // uses sequential addressing shared mem


////////////////////////////////////////////////////////////////////////////////
//
// MAIN
int main()
{
    // create array of vals
    int vals[SIZE], *vals_d;

    // create rng
    unsigned int seed = system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> dist(0, 1000);

    // init vals
    for (int i = 0; i < SIZE; ++i) vals[i] = dist(generator);

    /*
    // print vals
    for (int i = 0; i < SIZE; ++i)
        cout << vals[i] << ' ';
    cout << endl;
    */

    // copy vals to device memory
    int valMem = sizeof(int) * SIZE;
    cudaMalloc((void**)&vals_d, valMem);
    cudaMemcpy(vals_d, vals, valMem, cudaMemcpyHostToDevice);

    // call funcs
    //globalMax<<<1, SIZE / 2>>>(vals_d);
    //interleavingMax<<<1, SIZE>>>(vals_d);
    sequentialMax<<<1, SIZE>>>(vals_d);

    // copy device memory back to host
    cudaMemcpy(vals, vals_d, valMem, cudaMemcpyDeviceToHost);

    // print max
    //cout << vals[0] << endl;

    // free all device memory
    cudaFree(vals_d);
}

////////////////////////////////////////////////////////////////////////////////
//
// KERNEL functions
////////////////////////////////////////
// global memory implementation
__global__ void globalMax(int *vals)
{
    // thread index
    unsigned int idx = threadIdx.x;

    // reduction algorithm
    unsigned int elemIdx;
    for (unsigned int i = 1; i < SIZE; i *= 2)
    {
        elemIdx = idx * i * 2;

        if (elemIdx < SIZE)
            vals[elemIdx] = vals[elemIdx] < vals[elemIdx + i] ? 
                            vals[elemIdx + i] : vals[elemIdx];

        __syncthreads();
    }
}

////////////////////////////////////////
// interleaving addressing shared memory implementation
__global__ void interleavingMax(int *vals)
{
    // create shared val array
    static __shared__ int vals_s[SIZE];

    // thread index
    unsigned int idx = threadIdx.x;

    // load 1 element in per thread
    vals_s[idx] = vals[idx];
    __syncthreads();

    // reduction algorithm
    unsigned int elemIdx;
    for (unsigned int i = 1; i < SIZE; i *= 2)
    {
        elemIdx = idx * i * 2;

        if (elemIdx < SIZE)
            vals_s[elemIdx] = vals_s[elemIdx] < vals_s[elemIdx + i] ? 
                              vals_s[elemIdx + i] : vals_s[elemIdx];

        __syncthreads();
    }

    // transfer max val to global mem
    if (idx == 0) vals[0] = vals_s[0];
}

////////////////////////////////////////
// sequential addressing shared memory implementation
__global__ void sequentialMax(int *vals)
{
    // create shared val array
    static __shared__ int vals_s[SIZE];

    // thread index
    unsigned int idx = threadIdx.x;

    // load 1 element in per thread
    vals_s[idx] = vals[idx];
    __syncthreads();

    // reduction algorithm
    for (unsigned int i = SIZE / 2; i > 0; i /= 2)
    {
        if (idx < i)
            vals_s[idx] = vals_s[idx] < vals_s[idx + i] ? 
                          vals_s[idx + i] : vals_s[idx];

        __syncthreads();
    }

    // transfer max val to global mem
    if (idx == 0) vals[0] = vals_s[0];
}