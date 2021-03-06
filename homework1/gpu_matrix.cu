////////////////////////////////////////////////////////////////////////////////
//
// FILE:        gpu_matrix.cu
// DESCRIPTION: calculates matrix multiplication on gpu
// AUTHOR:      Dan Fabian
// DATE:        1/26/2020

#include <iostream>
#include <random>
#include <chrono>

using std::cout; using std::endl;
using namespace std::chrono;

// create matrices of size SIZExSIZE
const int SIZE = 128;

// kernel multiply func
__global__ void multiply(bool *a, bool *b, int *c)
{
	int xIdx = blockIdx.x, yIdx = threadIdx.x;
	for (int i = 0; i < SIZE; ++i)
		c[xIdx * SIZE + yIdx] += a[xIdx * SIZE + i] * b[i * SIZE + yIdx];	
}

int main()
{
	// create matrices
	bool a[SIZE][SIZE], b[SIZE][SIZE], *ad, *bd;
	int c[SIZE][SIZE], *cd;

	// create rng
	unsigned int seed = system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_int_distribution<int> dist(0, 1);

	// init matrices
	for (int i = 0; i < SIZE; ++i) for (int j = 0; j < SIZE; ++j)
	{
		a[i][j] = dist(generator);
		b[i][j] = dist(generator);
		c[i][j] = 0;
	}

	// memory size
	int boolSize = SIZE * SIZE * sizeof(bool);
	int intSize = SIZE * SIZE * sizeof(int);

	// allocate memory on device
	cudaMalloc((void**)&ad, boolSize);
	cudaMalloc((void**)&bd, boolSize);
	cudaMalloc((void**)&cd, intSize);

	// copy a and b to device memory
	cudaMemcpy(ad, a, boolSize, cudaMemcpyHostToDevice);
	cudaMemcpy(bd, b, boolSize, cudaMemcpyHostToDevice);

	// call multiply func
	multiply<<<SIZE, SIZE>>>(ad, bd, cd);

	// copy device memory back to host
	cudaMemcpy(c, cd, intSize, cudaMemcpyDeviceToHost);

	// print out final matrix
	for (int i = 0; i < SIZE; ++i) 
	{
		for (int j = 0; j < SIZE; ++j)
			cout << c[i][j] << ' ';
		cout << endl;
	}

	// free all device memory
	cudaFree(ad); cudaFree(bd); cudaFree(cd);

	return 0;
}
