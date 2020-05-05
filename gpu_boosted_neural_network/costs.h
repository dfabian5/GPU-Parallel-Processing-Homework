#ifndef COSTS_H
#define COSTS_H

////////////////////////////////////////////////////////////////////////////////
//
// FILE:        costs.h
// DESCRIPTION: contains cost objects for use in network object 
// AUTHOR:      Dan Fabian
// DATE:        3/13/2020

namespace Costs
{
    ////////////////////////////////////////
    // CROSS ENTROPY
    struct Cross_Entropy {
        static __device__ float func(const float& x, const float& y) 
        {
            return y * (1.0f - x) - x * (1.0f - y);
        }
    };

    ////////////////////////////////////////
    // SQUARED ERROR
    struct Squared_Error {
        static __device__ float func(const float& x, const float& y)
        {
            return 2.0f * (x - y);
        }
    };
}

#endif // COSTS_H