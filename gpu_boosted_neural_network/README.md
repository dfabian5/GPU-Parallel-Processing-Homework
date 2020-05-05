# Compile and Run test.cu
Inside test.cu it creates sample training data to model sin(x) for x in [-2, 2].
1. Inside the gpu_boosted_neural_network folder run `make test` to compile the project
2. After it has been compiled, run `./test`

# Report
## Analysis of Algorithms
1. The main two algorithms are the feedforward algorithm and the backpropagation algorithm. Both of these can be easily parallelized since they simplify down to vector and matrix operations.
2. Both algorithms were easily parallelizable.
3. The backpropagation algorithm is the performance bottleneck since there are many more operations that need to be performed
## Parallelizing Strategies
I used cuBLAS to help parallelize matrix multiplication and dot products, and for the rest I wrote gpu kernel functions to perform vector-vector operations.
## CPU vs GPU Algorithm Design
1. For parallelizing alorithms, used cuBLAS for the majority of the algorithms and I wrote two gpu kernels for handling unary and binary vector operations.
2. If I also used the cpu to help speed up computations I would split part of the vector and matrix computations between gpu and cpu, so less data had to be transfered between host and device memory which is the slowest part of using the gpu.
3. If I was only allowed to use a gpu I would limit my data transfers and compute as much as possible within a single kernel function.
## Details of Your Designs
1. I created a mini tensorflow-like api for constructing, training, and using neural networks to model functions and predict classes. I made seperate functors for each loss and activation function. I used polymorphism to create networks with varying activation functions as well as sizes.
2. I attempted to create multiple types of layers to be used in networks but fell short of time.
## Theoretical Analysis on Your Design
1. My design is memory bound. Most of the time spent is for transfering data to the gpu to run operations on.
2. In the ideal case, matrix multiplication would take O(n^2) time assuming one thread per element in the output matrix. Then the feedforward algorithm would take O(mn^2) assuming a network with m layers have the same size n.
## Performance Results
1. Approximately 500 GFLOPS per second
2. Approximately 100 GB/s bandwidth
3. Since cuBLAS scales quite well, GFLOPS increases up to 1200 GFLOPS for matrix multiplication and dot products as layer sizes increase.
