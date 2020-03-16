////////////////////////////////////////////////////////////////////////////////
//
// FILE:        one_dim_convolution.cu
// DESCRIPTION: implements one-dimensional convolution in 3 ways
// AUTHOR:      Dan Fabian
// DATE:        3/15/2020

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

using std::cout; using std::endl; using std::cin;

// constant size params
const int SIZE = 15000, MASK_SIZE = 1001, BLOCK_SIZE = 1024;

// gpu prototypes
__global__ void naive_conv(float *input, float *kernel, float *output, 
                           int maskSize, int size); // naive implementation
__global__ void const_mem_conv(float *input, float *output, 
                               int maskSize, int size); // uses constant memory
__global__ void shared_mem_conv(float *input, float *output, 
                                int maskSize, int size); // uses shared and constant mem

// device constant used in const_mem_conv and shared_mem_conv
__constant__ float kernel_c[MASK_SIZE];

////////////////////////////////////////////////////////////////////////////////
//
// MAIN
int main()
{
    // create arrays
    float *input_d, input[SIZE], *kernel_d, kernel[MASK_SIZE], *output_d, output[SIZE];

    // init kernel
    for (int i = 0; i < MASK_SIZE; ++i)
        if (i == MASK_SIZE / 2)
            kernel[i] = 1;
        else
            kernel[i] = 0;

    // init input array
    for (int i = 0; i < SIZE; ++i)
        input[i] = 1;

    // memory sizes to allocate
    int inOutMem = SIZE * sizeof(float), kernelMem = MASK_SIZE * sizeof(float);

    // determine which version will be ran before allocating mem
    cout << "Which version ('n': naive, 'c': constant mem, 's': shared mem): ";
    char c; cin >> c;

    // allocate memory on device
    cudaMalloc((void**)&input_d, inOutMem);
    if (c == 'n')
        cudaMalloc((void**)&kernel_d, kernelMem);
    cudaMalloc((void**)&output_d, inOutMem);

    // copy frequency and sorted arrays to device
    cudaMemcpy(input_d, input, inOutMem, cudaMemcpyHostToDevice);
    if (c == 'n')
        cudaMemcpy(kernel_d, kernel, kernelMem, cudaMemcpyHostToDevice);

    // copy kernel to constant memory
    if (c != 'n')
        cudaMemcpyToSymbol(kernel_c, kernel, kernelMem);

    // call gpu func
    int gridSize = ceil(float(SIZE) / float(BLOCK_SIZE));

    if (c == 'n')
        naive_conv<<<gridSize, BLOCK_SIZE>>>(input_d, kernel_d, output_d, MASK_SIZE, SIZE);
    else if (c == 'c')
        const_mem_conv<<<gridSize, BLOCK_SIZE>>>(input_d, output_d, MASK_SIZE, SIZE);
    else if (c == 's')
        shared_mem_conv<<<gridSize, BLOCK_SIZE>>>(input_d, output_d, MASK_SIZE, SIZE);

    // copy device memory back to host
    cudaMemcpy(output, output_d, inOutMem, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(input_d); cudaFree(kernel_d); cudaFree(output_d);

    // print result
    for (int i = 0; i < SIZE; ++i)
        cout << output[i] << ' ';
    cout << endl;

    cudaDeviceSynchronize();
}

////////////////////////////////////////////////////////////////////////////////
//
// KERNEL functions
////////////////////////////////////////
// naive implementation
__global__ void naive_conv(float *input, float *kernel, float *output, 
                           int maskSize, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x, 
        start = idx - (maskSize / 2);

    float val = 0;
    for (int i = 0; i < maskSize; ++i)
        if (0 <= start + i && start + i < size)
            val += input[start + i] * kernel[i];

    // copy to global mem
    if (idx < size)
        output[idx] = val;
}

////////////////////////////////////////
// faster implementation that uses constant memory
__global__ void const_mem_conv(float *input, float *output, 
                               int maskSize, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x, 
        start = idx - (maskSize / 2);

    float val = 0;
    for (int i = 0; i < maskSize; ++i)
        if (0 <= start + i && start + i < size)
            val += input[start + i] * kernel_c[i];

    // copy to global mem
    if (idx < size)
        output[idx] = val;
}

////////////////////////////////////////
// implementation using shared and constant memory
__global__ void shared_mem_conv(float *input, float *output, 
                                int maskSize, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x,
        lowerBound = blockDim.x * blockIdx.x,
        upperBound = blockDim.x * (blockIdx.x + 1),
        start = idx - (maskSize / 2);

    // shared mem
    __shared__ float tile_s[BLOCK_SIZE];

    // load elem
    if (idx < size)
        tile_s[threadIdx.x] = input[idx];
    __syncthreads();

    float val = 0;
    bool bound;
    for (int i = 0, elem = start; i < maskSize; ++i, ++elem)
    {
        // check if its within array
        bound = 0 <= elem && elem < size;

        // if its within bounds, use shared mem
        if (lowerBound <= elem && elem < upperBound && bound)
            val += tile_s[elem - lowerBound] * kernel_c[i];       
        // resort to global mem if needed
        else if (bound)
            val += input[elem] * kernel_c[i];
    }

    // copy to global mem
    if (idx < size)
        output[idx] = val;
}