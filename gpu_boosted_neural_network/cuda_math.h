#ifndef CUDA_MATH_H
#define CUDA_MATH_H

////////////////////////////////////////////////////////////////////////////////
//
// FILE:        cuda_math.h
// DESCRIPTION: contains implementations for all gpu and gpu calling host functions
// AUTHOR:      Dan Fabian
// DATE:        3/13/2020

#include <cuda.h>
#include <cuda_runtime.h>
#include <stddef.h> // for size_t
#include <math.h>
#include <cublas_v2.h>

////////////////////////////////////////////////////////////////////////////////
//
// APPLY_UNARY_FUNC
////////////////////////////////////////
// device function to apply a single argument function to each elem
template <typename F>
__global__ void apply_unary_func_device(float *x, float *result, 
                                        const size_t size, F func)
{
    // element idx
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    // check thread is within array bounds
    if (idx < size)
        result[idx] = func(x[idx]);
}

////////////////////////////////////////
// host function to allocate memory and call gpu func for unary function
template <typename F>
float* apply_unary_func(float *x, const size_t& size, F func)
{
    // get device properties
    cudaDeviceProp DEVICE_PROP;
    cudaGetDeviceProperties(&DEVICE_PROP, 0); // assuming one gpu

    // create arrays for device
    float *x_d, *result, *result_d;
    result = new float[size];

    // memory sizes to allocate
    int memory = sizeof(float) * size;

    // allocate memory on device
    cudaMalloc((void**)&x_d, memory);
    cudaMalloc((void**)&result_d, memory);

    // copy arrays to device
    cudaMemcpy(x_d, x, memory, cudaMemcpyHostToDevice);

    // call gpu func
    int blockSize = DEVICE_PROP.maxThreadsPerBlock;
    int gridSize = ceil(float(size) / float(blockSize));
    apply_unary_func_device<F><<<gridSize, blockSize>>>(x_d, result_d, 
                                                        size, func);

    // copy device memory back to host
    cudaMemcpy(result, result_d, memory, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(x_d); cudaFree(result_d);

    return result;
}

////////////////////////////////////////////////////////////////////////////////
//
// APPLY_BINARY_FUNC
////////////////////////////////////////
// device function to get the cost values from cost function
template <typename F>
__global__ void apply_binary_func_device(float *x, float *y, float *result, 
                                         const size_t size, F func)
{
    // element idx
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    // check thread is within array bounds
    if (idx < size)
        result[idx] = func(x[idx], y[idx]);
}

////////////////////////////////////////
// host function to allocate memory and call gpu func
template <typename F>
float* apply_binary_func(float *x, float *y, const size_t& size, F func)
{
    // get device properties
    cudaDeviceProp DEVICE_PROP;
    cudaGetDeviceProperties(&DEVICE_PROP, 0); // assuming one gpu

    // create arrays for device
    float *x_d, *y_d, *result, *result_d;
    result = new float[size];

    // memory sizes to allocate
    int memory = sizeof(float) * size;

    // allocate memory on device
    cudaMalloc((void**)&x_d, memory);
    cudaMalloc((void**)&y_d, memory);
    cudaMalloc((void**)&result_d, memory);

    // copy arrays to device
    cudaMemcpy(x_d, x, memory, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, memory, cudaMemcpyHostToDevice);

    // call gpu func
    int blockSize = DEVICE_PROP.maxThreadsPerBlock;
    int gridSize = ceil(float(size) / float(blockSize));
    apply_binary_func_device<F><<<gridSize, blockSize>>>(x_d, y_d, result_d, 
                                                         size, func);

    // copy device memory back to host
    cudaMemcpy(result, result_d, memory, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(x_d); cudaFree(y_d); cudaFree(result_d);

    return result;
}

#endif // CUDA_MATH_H