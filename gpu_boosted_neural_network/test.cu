#include "cuda_math.h"
#include "activations.h"
#include "layers.h"
#include "network.h"
#include "costs.h"
#include <iostream>

using std::cout; using std::endl;

int main()
{
    // set up network
    Network<Costs::Cross_Entropy> network({new Layers::Dense<Activations::Linear>(1),
                                           new Layers::Dense<Activations::Linear>(50),
                                           new Layers::Dense<Activations::Linear>(1)});

    // create rng
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
    auto rand = std::bind(distribution, generator);

    // set up data
    vector<vector<float>> xData;
    vector<vector<float>> yData;
    for (int i = 0; i < 10000; ++i)
    {
        vector<float> x(1, rand());
        xData.push_back(x);

        vector<float> y(1, sin(x[0]));
        yData.push_back(y);
    }

    // test
    cout << "Test Before Training:" << endl;
    float loss = 0;
    for (int i = 0; i < 10; ++i)
    {
        float input = rand();
        vector<float> x = network.feed_forward(vector<float>(1, input));
        cout << "Input: " << input << " Prediction: " << x[0] << " Correct Answer: " << sin(input) << endl;
        loss += abs(x[0] - sin(input));
    }
    cout << "Loss: " << loss << endl;

    // train
    network.train(xData, yData, 200);

    // test
    cout << "Test After Training:" << endl;
    loss = 0;
    for (int i = 0; i < 10; ++i)
    {
        float input = rand();
        vector<float> x = network.feed_forward(vector<float>(1, input));
        cout << "Input: " << input << " Prediction: " << x[0] << " Correct Answer: " << sin(input) << endl;
        loss += abs(x[0] - sin(input));
    }
    cout << "Loss: " << loss << endl;

    cudaDeviceSynchronize();
}