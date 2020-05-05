#ifndef NETWORK_H
#define NETWORK_H

////////////////////////////////////////////////////////////////////////////////
//
// FILE:        network.h
// DESCRIPTION: contains network object and implementation
// AUTHOR:      Dan Fabian
// DATE:        3/25/2020

#include "layers.h"
#include <vector>
#include <iostream>
#include <random>

using std::vector;
using std::cout; using std::endl;

////////////////////////////////////////////////////////////////////////////////
//
// NETWORK
template <typename C>
class Network {
public:
    // constructor
    Network(vector<Layers::Layer*> layers, const float& step = 0.1f, const float& lambda = 0.0f);

    // methods
    vector<float> feed_forward (vector<float> input);                           // feeds input through network
    void backpropagate         (float *alpha, float *y, const size_t& setSize); // adjusts weights and biases of layers
    void train                 (vector<vector<float>> x, vector<vector<float>> y, 
                                const size_t& epochs);                          // trains network

    // accessors
    void setStep(const float& step) { step_ = step; }
    void setLambda(const float& lambda) { lambda_ = lambda; }
    float getStep() const { return step_; }
    float getLambda() const { return lambda_; }

private:
    // data
    vector<Layers::Layer*> layers_;
    float step_;
    float lambda_;

    // device prop
    size_t maxThreadBlock_;
};

////////////////////////////////////////////////////////////////////////////////
//
// NETWORK functions
////////////////////////////////////////
// constructor, connect and initialize all layers
template <typename C>
Network<C>::Network(vector<Layers::Layer*> layers, const float& step, 
                    const float& lambda) :
    layers_(layers),
    step_(step),
    lambda_(lambda)
{
    for (int i = 1; i < layers_.size(); ++i)
        layers_[i]->connect_to_prev(layers_[i - 1]->getNeurons());


    // device query
    cudaDeviceProp DEVICE_PROP;
    cudaGetDeviceProperties(&DEVICE_PROP, 0); // assuming one gpu

    maxThreadBlock_ = DEVICE_PROP.maxThreadsPerBlock;
}

////////////////////////////////////////
// feeds input throught network and returns final output
template <typename C>
vector<float> Network<C>::feed_forward(vector<float> input)
{
    // convert vector to array
    float *alpha = input.data();

    // set z for input layer
    layers_.front()->setZ(alpha);

    // create handle for cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    // now feed through layers
    for (int i = 1; i < layers_.size(); ++i)
        alpha = layers_[i]->feed_forward(handle, maxThreadBlock_, alpha);

    // destroy cublas handle
    cublasDestroy(handle);

    // convert output back to vector and return
    vector<float> out;
    out.assign(alpha, alpha + layers_.back()->getNeurons());
    return out;
}

////////////////////////////////////////
// adjusts weights and biases in network, helper function for training
template <typename C>
void Network<C>::backpropagate(float *alpha, float *y, const size_t& setSize)
{
    // get output layer deltas
    float *delta = apply_binary_func(alpha, y, layers_.back()->getNeurons(), 
                                     [] __device__ (const float& a, const float& b) { return C::func(a, b); });
    layers_.back()->setDeltas(delta);

    // create handle for cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    // find deltas for all layers
    for (int i = layers_.size() - 2; i > 0; --i)
        layers_[i]->find_deltas(handle,
                                maxThreadBlock_,
                                layers_[i + 1]->getWeights(), 
                                layers_[i + 1]->getNeurons(), 
                                layers_[i + 1]->getDeltas());

    // destroy cublas handle
    cublasDestroy(handle);

    // adjust weights and biases
    for (int i = 1; i < layers_.size(); ++i)
        layers_[i]->adjust_params(maxThreadBlock_, layers_[i - 1]->getZ(), 
                                  step_, lambda_, setSize, i == 1);
}

////////////////////////////////////////
// trains network
template <typename C>
void Network<C>::train(vector<vector<float>> x, vector<vector<float>> y, const size_t& epochs)
{
    std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(0, x.size() - 1);
	auto rand = std::bind(distribution, generator);

    cout << "Training..." << endl << "Iteration ";
    for (int i = 0; i < epochs; ++i)
    {
        size_t index = rand();
        vector<float> output = feed_forward(x[index]);

        backpropagate(output.data(), y[index].data(), 1);
        cout << i << ", ";
    }
    cout << endl;
}

#endif // NETWORK_H