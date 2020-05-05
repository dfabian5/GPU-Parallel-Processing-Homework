#ifndef LAYERS_H
#define LAYERS_H

////////////////////////////////////////////////////////////////////////////////
//
// FILE:        layers.h
// DESCRIPTION: contains templated layer objects for use in network 
// AUTHOR:      Dan Fabian
// DATE:        3/13/2020

#include <random>
#include "cuda_math.h"
#include <iostream>

using std::vector;
using std::cout; using std::endl;

////////////////////////////////////////////////////////////////////////////////
//
// LAYERS
namespace Layers
{
    ////////////////////////////////////////
    // LAYER base class
    class Layer {
    public:
        // methods
        virtual void   connect_to_prev (const size_t& prev) = 0;
        virtual float* feed_forward    (const cublasHandle_t& handle, 
                                        const size_t maxThreadBlock, 
                                        float *prevActivations) = 0;
        virtual void   find_deltas     (const cublasHandle_t& handle, 
                                        const size_t maxThreadBlock, 
                                        float *nextLayerWeights, 
                                        const size_t& nextLayerNeurons, 
                                        float *nextLayerDeltas) = 0;
        virtual void   adjust_params   (const size_t maxThreadBlock, 
                                        float *prevLayerZ, const float& step, 
                                        const float& lambda, const size_t& setSize, 
                                        const bool& firstLayer) = 0;

        // accessors
        virtual size_t getNeurons    () const       = 0;
        virtual float* getWeights    ()             = 0;
        virtual float* getDeltas     ()             = 0;
        virtual float* getZ          ()             = 0;
        virtual void   setDeltas     (float *delta) = 0;   
        virtual void   setZ          (float *z)     = 0;
    };

    ////////////////////////////////////////
    // DENSE derived class
    template <typename A>
    class Dense: public Layer {
    public:
        // constructor
        Dense (const size_t& neurons) : neurons_(neurons) {}

        // methods
        void   connect_to_prev (const size_t& prev);                                     // connect layer weights to previous layer size and init
        float* feed_forward    (const cublasHandle_t& handle, 
                                const size_t maxThreadBlock, float *prevActivations);    // compute activations from neurons in layer
        void   find_deltas     (const cublasHandle_t& handle, 
                                const size_t maxThreadBlock, float *nextLayerWeights, 
                                const size_t& nextLayerNeurons, float *nextLayerDeltas); // finds deltas for backprop alg
        void   adjust_params   (const size_t maxThreadBlock, float *prevLayerZ, 
                                const float& step, const float& lambda, 
                                const size_t& setSize, const bool& firstLayer);          // use after deltas have been found in backprop

        // accessors
        size_t getNeurons    () const       { return neurons_; }
        float* getWeights    ()             { return weights_; }
        float* getDeltas     ()             { return delta_; }
        float* getZ          ()             { return z_; }
        void   setDeltas     (float *delta) { delta_ = delta; }
        void   setZ          (float *z)     { z_ = z; }

    private:
        float* weights_;
        float* biases_;
        float* z_;           // value of neuron before activation function applied, used in backprop alg
        float* delta_;       // used in backprop
        size_t prevNeurons_;
        size_t neurons_;
    };
}

////////////////////////////////////////////////////////////////////////////////
//
// DENSE functions
// note: weights are indexed as       
//        L1                W[0][0]         L2
//        neuron -------------------------- neuron
//               |          W[1][0]
//               -------------------------- neuron
////////////////////////////////////////
// connects layer weights to previous layer and initializes weights and biases
template <typename A>
void Layers::Dense<A>::connect_to_prev(const size_t& prev)
{
    // set previous layer neuron count
    prevNeurons_ = prev;

    // create weights and biases arrays
    weights_ = new float[prevNeurons_ * neurons_];
    biases_ = new float[neurons_];

    // init seed and create distibution
    std::default_random_engine generator;
    std::normal_distribution<float> weightDist(0, (1.0 / sqrt(prevNeurons_))); // for weights
    std::normal_distribution<float> biasDist(0, 1); // for biases

    // init weights with normal distribution with a mean of 0 and SD of 1/sqrt(incoming weights)
    for (size_t i = 0; i != neurons_; ++i)
        for (size_t j = 0; j != prevNeurons_; ++j)
            weights_[i * prevNeurons_ + j] = weightDist(generator);

    // init biases
    for (size_t i = 0; i != neurons_; ++i)
        biases_[i] = biasDist(generator);
}

////////////////////////////////////////
// feeds previous layer activations to get new activations
template <typename A>
float* Layers::Dense<A>::feed_forward(const cublasHandle_t& handle, 
                                      const size_t maxThreadBlock, 
                                      float *prevActivations)
{
    // create arrays for device
    float *weights_d, *prevActivations_d, *biases_d, *result = new float[neurons_ * prevNeurons_];

    // memory sizes to allocate, neurons_ = height of matrix, prevNeurons_ = width
    int matrixMemory = sizeof(float) * neurons_ * prevNeurons_, 
        vecMemory = sizeof(float) * prevNeurons_,
        resultMemory = sizeof(float) * neurons_;

    // allocate memory on device
    cudaMalloc((void**)&weights_d, matrixMemory);
    cudaMalloc((void**)&prevActivations_d, vecMemory);
    cudaMalloc((void**)&biases_d, resultMemory);
    
    // copy arrays to device
    cudaMemcpy(weights_d, weights_, matrixMemory, cudaMemcpyHostToDevice);
    cudaMemcpy(prevActivations_d, prevActivations, vecMemory, cudaMemcpyHostToDevice);
    cudaMemcpy(biases_d, biases_, resultMemory, cudaMemcpyHostToDevice);

    // prepare args
    const float *weights_d_c = weights_d, *prevActivations_d_c = prevActivations_d,
                alpha = 1.0f, beta = 1.0f;
    
    // alpha * weights * prevActivations + beta * biases -> biases
    cublasSgemv(handle, CUBLAS_OP_T, neurons_, prevNeurons_, &alpha,
                weights_d_c, neurons_, prevActivations_d_c, 1, &beta, biases_d, 1);

    // copy device memory back to host
    cudaMemcpy(result, biases_d, resultMemory, cudaMemcpyDeviceToHost);

    // free memory, keep biases in mem for activations
    cudaFree(weights_d); cudaFree(prevActivations_d);

    // store intermediate results for backprop
    z_ = result;

    // apply activation
    int gridSize = ceil(float(neurons_) / float(maxThreadBlock));
    apply_unary_func_device<<<gridSize, maxThreadBlock>>>(biases_d, biases_d, neurons_,
                                                     [] __device__ (const float& a) { return A::activate(a); });
                                                    
    // copy device memory back to host
    cudaMemcpy(result, biases_d, resultMemory, cudaMemcpyDeviceToHost);

    // free last bit of memory
    cudaFree(biases_d);

    return result;
}

////////////////////////////////////////
// calculates deltas in layer to adjust weights and biases
template <typename A>
void Layers::Dense<A>::find_deltas(const cublasHandle_t& handle,
                                   const size_t maxThreadBlock, 
                                   float *nextLayerWeights, 
                                   const size_t& nextLayerNeurons, 
                                   float *nextLayerDeltas)
{
    // create arrays for device
    float *nextLayerWeights_d, *nextLayerDeltas_d;

    // memory sizes to allocate
    int matrixMemory = sizeof(float) * neurons_ * nextLayerNeurons,
        deltaMemory = sizeof(float) * nextLayerNeurons;

    // allocate memory on device
    cudaMalloc((void**)&nextLayerWeights_d, matrixMemory);
    cudaMalloc((void**)&nextLayerDeltas_d, deltaMemory);

    // copy arrays to device
    cudaMemcpy(nextLayerWeights_d, nextLayerWeights, matrixMemory, cudaMemcpyHostToDevice);
    cudaMemcpy(nextLayerDeltas_d, nextLayerDeltas, deltaMemory, cudaMemcpyHostToDevice);

    // init delta array
    delta_ = new float[neurons_];

    // compute each delta element with dot product
    const float *nextLayerWeights_d_c = nextLayerWeights_d, 
                *nextLayerDeltas_d_c = nextLayerDeltas_d;
    for (int i = 0; i < neurons_; ++i)
        cublasSdot(handle, nextLayerNeurons, &nextLayerWeights_d_c[i], neurons_,
                   nextLayerDeltas_d_c, 1, &delta_[i]);
                   
    // free memory
    cudaFree(nextLayerWeights_d); cudaFree(nextLayerDeltas_d);
    
    // create arrays for device
    float *z_d, *delta_d;

    // memory sizes to allocate
    int vecMem = sizeof(float) * neurons_;

    // allocate memory on device
    cudaMalloc((void**)&z_d, vecMem);
    cudaMalloc((void**)&delta_d, vecMem);

    // copy arrays to device
    cudaMemcpy(z_d, z_, vecMem, cudaMemcpyHostToDevice);
    cudaMemcpy(delta_d, delta_, vecMem, cudaMemcpyHostToDevice);

    // get grid size
    int gridSize = ceil(float(neurons_) / float(maxThreadBlock));

    // get primes
    apply_unary_func_device<<<gridSize, maxThreadBlock>>>(z_d, z_d, neurons_,
                                                          [] __device__ (const float& a) { return A::prime(a); });

    // now multiply by the derivatives
    apply_binary_func_device<<<gridSize, maxThreadBlock>>>(delta_d, z_d, delta_d, neurons_,
                                                           [] __device__ (const float& a, const float& b) { return a * b; });

    // copy memory back to host
    cudaMemcpy(delta_, delta_d, vecMem, cudaMemcpyDeviceToHost);
    
    // free memory
    cudaFree(delta_d); cudaFree(z_d);
}

////////////////////////////////////////
// after deltas in all layers are found, this adjusts weights and biases
template <typename A>
void Layers::Dense<A>::adjust_params(const size_t maxThreadBlock, 
                                     float *prevLayerZ, const float& step, 
                                     const float& lambda, const size_t& setSize, 
                                     const bool& firstLayer)
{
    // create arrays for device
    float *prevLayerZ_d, *delta_d, *bias_d, *weights_d, *stepDelta_d, *regularization_d,
          *adjust_d;

    // memory sizes to allocate
    int prevLayerZMemory = sizeof(float) * prevNeurons_,
        deltaMemory = sizeof(float) * neurons_,
        weightsMemory = sizeof(float) * prevNeurons_ * neurons_;

    // allocate memory on device
    cudaMalloc((void**)&prevLayerZ_d, prevLayerZMemory);
    cudaMalloc((void**)&delta_d, deltaMemory);
    cudaMalloc((void**)&bias_d, deltaMemory);
    cudaMalloc((void**)&weights_d, weightsMemory);
    cudaMalloc((void**)&stepDelta_d, deltaMemory);
    cudaMalloc((void**)&regularization_d, weightsMemory);
    cudaMalloc((void**)&adjust_d, prevLayerZMemory);

    // copy arrays to device
    cudaMemcpy(prevLayerZ_d, prevLayerZ, prevLayerZMemory, cudaMemcpyHostToDevice);
    cudaMemcpy(delta_d, delta_, deltaMemory, cudaMemcpyHostToDevice);
    cudaMemcpy(bias_d, biases_, deltaMemory, cudaMemcpyHostToDevice);
    cudaMemcpy(weights_d, weights_, deltaMemory, cudaMemcpyHostToDevice);

    // get grid size
    int gridSize = ceil(float(neurons_) / float(maxThreadBlock));

    // adjust biases
    // multiply step constant 
    apply_unary_func_device<<<gridSize, maxThreadBlock>>>(delta_d, stepDelta_d, neurons_,
                                                          [=] __device__ (const float& a) { return a * step; });

    // add to biases
    apply_binary_func_device<<<gridSize, maxThreadBlock>>>(bias_d, stepDelta_d, bias_d, neurons_,
                                                           [] __device__ (const float& a, const float& b) { return a + b; });

    // copy memory back to host and free bias memory
    cudaMemcpy(biases_, bias_d, deltaMemory, cudaMemcpyDeviceToHost);
    cudaFree(bias_d);

    // adjust weights
    // get previous layer activation
    if (!firstLayer) 
        apply_unary_func_device<<<gridSize, maxThreadBlock>>>(prevLayerZ_d, prevLayerZ_d, prevNeurons_,
            [] __device__ (const float& a) { return A::activate(a); });

    // get regularization terms
    apply_unary_func_device<<<gridSize, maxThreadBlock>>>(weights_d, regularization_d, prevNeurons_ * neurons_,
            [=] __device__ (const float& a) { return a * (lambda / setSize); });
        
    // copy memory back to host and free
    float *stepDelta = new float[neurons_];
    cudaMemcpy(stepDelta, stepDelta_d, deltaMemory, cudaMemcpyDeviceToHost);
    cudaFree(stepDelta_d);

    // for each neuron, multiply stepDelta with prevLayerActivation, then adjust neuron weights
    for (int i = 0; i < neurons_; ++i)
    {
        // get amount to adjust weights by
        float tmp = stepDelta[i];

        apply_unary_func_device<<<gridSize, maxThreadBlock>>>(prevLayerZ_d, adjust_d, prevNeurons_,
            [=] __device__ (const float& a) { return a * tmp; });
        
        apply_binary_func_device<<<gridSize, maxThreadBlock>>>(adjust_d, &regularization_d[i * prevNeurons_], adjust_d, prevNeurons_,
            [] __device__ (const float& a, const float& b) { return a + b; });
        
        // add adjustment to current weights
        apply_binary_func_device<<<gridSize, maxThreadBlock>>>(&weights_d[i * prevNeurons_], adjust_d, adjust_d, prevNeurons_,
            [] __device__ (const float& a, const float& b) { return a + b; });
    
        cudaMemcpy(&weights_[i * prevNeurons_], adjust_d, prevLayerZMemory, cudaMemcpyDeviceToHost);
    }

    // free all memory
    cudaFree(prevLayerZ_d); cudaFree(delta_d); cudaFree(weights_d);
    cudaFree(regularization_d); cudaFree(adjust_d);
}

#endif // LAYERS_H