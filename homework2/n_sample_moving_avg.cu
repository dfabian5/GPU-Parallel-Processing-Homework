////////////////////////////////////////////////////////////////////////////////
//
// FILE:        n_sample_moving_avg.cu
// DESCRIPTION: runs N Sample Moving Average Filtering algorithm on gpu
// AUTHOR:      Dan Fabian
// DATE:        2/16/2020

#include <iostream>
#include <random>
#include <chrono>

using std::cout; using std::endl; using std::cin;
using namespace std::chrono;

const int NUM_OF_VALS = 10000, N = 256, NUM_OF_AVG = NUM_OF_VALS - N + 1;

// kernal func
__global__ void movingAvg(int *vals, float *avg)
{
    // number of average calculations a single thread performs
    int avgCalcPerThread = ceilf(float(NUM_OF_AVG) / float(blockDim.x * gridDim.x));

    // thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // get first avg val for thread
    int avgIdx = idx * avgCalcPerThread;
    for (int i = 0; i < N && avgIdx < NUM_OF_AVG; ++i)
        avg[avgIdx] += vals[avgIdx + i];
    avg[avgIdx] /= N;

    // calculate the rest of avg vals for thread
    int maxAvgIdx = avgCalcPerThread * (idx + 1);
    for (avgIdx = idx * avgCalcPerThread + 1; 
         avgIdx < maxAvgIdx && avgIdx < NUM_OF_AVG; 
         ++avgIdx)
        avg[avgIdx] = (avg[avgIdx - 1] * N + vals[avgIdx + N - 1] - vals[avgIdx - 1]) / N;
}

int main()
{
    // ask user for grid and block dims, must multiply together to get NUM_OF_VALS
    cout << "Enter Grid X Dim: ";
    int gridDim; cin >> gridDim;

    cout << "Enter Block X Dim: ";
    int blockDim; cin >> blockDim;

    // create arrays of vals
    int vals[NUM_OF_VALS], *vals_d;
    float avg[NUM_OF_AVG], *avg_d;

    // create rng
	unsigned int seed = system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_int_distribution<int> dist(0, 5);

    // init vals
    for (int i = 0; i < NUM_OF_VALS; ++i)
    {
        vals[i] = dist(generator);

        if (i < NUM_OF_AVG)
            avg[i] = 0;
    }

    // memory sizes to allocate
    int valMem = sizeof(int) * NUM_OF_VALS, avgMem = sizeof(float) * NUM_OF_AVG;

    // allocate memory on device
    cudaMalloc((void**)&vals_d, valMem);
    cudaMalloc((void**)&avg_d, avgMem);
    
    // copy vals and avg to device
    cudaMemcpy(vals_d, vals, valMem, cudaMemcpyHostToDevice);
    cudaMemcpy(avg_d, avg, avgMem, cudaMemcpyHostToDevice);
    
    // call func
    movingAvg<<<gridDim, blockDim>>>(vals_d, avg_d);

    // copy device memory back to host
    cudaMemcpy(avg, avg_d, avgMem, cudaMemcpyDeviceToHost);

    /*
    // print vals
    for (int i = 0; i < NUM_OF_VALS; ++i)
        cout << vals[i] << ' ';
    cout << endl;

    // print averages
    for (int i = 0; i < NUM_OF_AVG; ++i)
        cout << avg[i] << ' ';
    cout << endl;
    */

    // free all device memory
    cudaFree(vals_d); cudaFree(avg_d); 
}