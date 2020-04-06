////////////////////////////////////////////////////////////////////////////////
//
// FILE:        burrows_wheeler_encoder.cu
// DESCRIPTION: uses bitonic sort to encode a string with BWT
// AUTHOR:      Dan Fabian
// DATE:        4/5/2020

#include <iostream>
#include <stdio.h>

using std::cout; using std::endl;

// constants, MUST BE A POWER OF 2 IN LENGTH
const int SIZE = 8;
const char STRING[] = "^BANANA|";

// kernal func prototype
__global__ void bitonic_sort(char *string, int *indices);

////////////////////////////////////////////////////////////////////////////////
//
// MAIN
int main()
{
    // create array of vals
    char *string_d;
    int *indices = new int[SIZE], *indices_d;

    // copy string to device memory and allocate mem
    int stringMem = sizeof(char) * SIZE, indexMem = sizeof(int) * SIZE;
    cudaMalloc((void**)&string_d, stringMem);
    cudaMalloc((void**)&indices_d, indexMem);
    cudaMemcpy(string_d, STRING, stringMem, cudaMemcpyHostToDevice);
    cudaMemcpy(indices_d, indices, indexMem, cudaMemcpyHostToDevice);

    // sort
    bitonic_sort<<<1, SIZE>>>(string_d, indices_d);

    // copy device memory back to host
    cudaMemcpy(indices, indices_d, indexMem, cudaMemcpyDeviceToHost);

    // print out encoded string
    for (int i = 0; i < SIZE; ++i)
        if (indices[i] != 0)    
            cout << STRING[indices[i] - 1] << ' ';
        else    
            cout << STRING[SIZE - 1] << ' ';
    cout << endl;

    // free all device memory
    cudaFree(indices_d); cudaFree(string_d);

    cudaDeviceSynchronize();
}

////////////////////////////////////////////////////////////////////////////////
//
// KERNEL function
////////////////////////////////////////
// compare strings
__device__ bool lessThan(char *string, const int& pos1, const int& pos2, const int &size)
{
    int i = 0;
    while (string[(pos1 + i) % size] == string[(pos2 + i) % size] && i < size) ++i;

    if (i == size) return false;

    return string[(pos1 + i) % size] < string[(pos2 + i) % size];
}

////////////////////////////////////////
// gpu sort func
__global__ void bitonic_sort(char *string, int *indices)
{
    const int size = SIZE;

    // create shared arrays
    static __shared__ char string_s[size]; // holds original string
    static __shared__ int indices_s[size]; // holds char indices of sorted array

    // thread idx
    int idx = threadIdx.x;

    // load 1 elem in each array per index
    string_s[idx] = string[idx]; 
    indices_s[idx] = idx;

    // bitonic sort alg
    int tmp, elemIdx1, elemIdx2, strIdx1, strIdx2;
    bool max; // if max then put max elem in higher index
    for (int i = 2; i <= size; i *= 2) 
    {
        // bitonic merge of size i
        max = (idx % i) < (i / 2);
        for (int j = i / 2; j > 0; j /= 2)
        {
            // get element indices to compare
            elemIdx1 = (idx / j) * (j * 2) + idx % j;
            elemIdx2 = elemIdx1 + j;

            strIdx1 = indices_s[elemIdx1];
            strIdx2 = indices_s[elemIdx2];

            // check if swap is needed
            if ((elemIdx2 < size) && 
                ((max && lessThan(string_s, strIdx2, strIdx1, size)) ||
                (!max && lessThan(string_s, strIdx1, strIdx2, size))))
            {
                // swap indices
                tmp = indices_s[elemIdx1];
                indices_s[elemIdx1] = indices_s[elemIdx2];
                indices_s[elemIdx2] = tmp;
            }

            // need to sync before next step
            __syncthreads();
        }
    }

    // transfer memory to global
    indices[idx] = indices_s[idx];
}