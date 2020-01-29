////////////////////////////////////////////////////////////////////////////////
//
// FILE:        cpu_matrix.cpp
// DESCRIPTION: calculates matrix multiplication on cpu
// AUTHOR:      Dan Fabian
// DATE:        1/26/2020

#include <iostream>
#include <random>
#include <chrono>

using std::cout; using std::endl;
using namespace std::chrono;

// create matrices of size SIZExSIZE
const int SIZE = 128;

typedef high_resolution_clock Clock;

void multiply(bool a[][SIZE], bool b[][SIZE], int c[][SIZE])
{
    for (int row = 0; row < SIZE; ++row) for (int col = 0; col < SIZE; ++col)
        for (int i = 0; i < SIZE; ++i)
            c[row][col] += a[row][i] * b[i][col];
}

int main()
{
    bool a[SIZE][SIZE], b[SIZE][SIZE];
    int c[SIZE][SIZE];

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

    // multiply
    auto ti = Clock::now();
    multiply(a, b, c);
    auto tf = Clock::now();

    // print out answer
    for (int i = 0; i < SIZE; ++i) 
    {
        for (int j = 0; j < SIZE; ++j)
            cout << c[i][j] << ' ';
        cout << endl;
    }

    // print out computation time
    cout << "Computation took "
		<< duration_cast<nanoseconds>(tf - ti).count()
		<< " nanooseconds" << endl;
}