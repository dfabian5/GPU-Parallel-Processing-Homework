#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

////////////////////////////////////////////////////////////////////////////////
//
// FILE:        activations.h
// DESCRIPTION: contains neuron activation objects for use in layer objects
// AUTHOR:      Dan Fabian
// DATE:        3/13/2020

#include <cuda.h>
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
//
// ACTIVATIONS
namespace Activations
{
    ////////////////////////////////////////
    // SIGMOID 
    struct Sigmoid {
        static __device__ float activate (const float& x) { return 1.0f / (1.0f + expf(-x)); }
        static __device__ float prime    (const float& x) { return activate(x) * (1.0f - activate(x)); }
    };

    ////////////////////////////////////////
    // RELU
    struct Relu {
        static __device__ float activate (const float& x) { return fmaxf(x, 0.0f); }
        static __device__ float prime    (const float& x) { return x <= 0.0f ? 0.0f : 1.0f; }
    };

    ////////////////////////////////////////
    // LINEAR
    struct Linear {
        static __device__ float activate (const float& x) { return x; }
        static __device__ float prime    (const float& x) { return 1.0f; }
    };
}

#endif // ACTIVATIONS_H