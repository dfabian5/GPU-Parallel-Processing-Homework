#ifndef ENDCODER_H
#define ENDCODER_H

////////////////////////////////////////////////////////////////////////////////
//
// FILE:        encoder.h
// DESCRIPTION: contains class and implementation for huffman coding a text file
// AUTHOR:      Dan Fabian
// DATE:        3/6/2020

#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <cuda.h>
#include <cuda_runtime.h>

using std::cout; using std::endl;
using std::ifstream; using std::ofstream;
using std::string;
using std::queue;

// number of symbols used, size of char
const int SIZE = 256;

// gpu function to sort frequencies
__global__ void bitonicSort(unsigned int *freq, unsigned int *sorted);

////////////////////////////////////////////////////////////////////////////////
//
// NODE
struct Node {
    Node(const unsigned char& c, const unsigned int& f) : 
        data_(c), freq_(f), left_(nullptr), right_(nullptr) {}

    unsigned char data_;
    unsigned int freq_;
    Node *left_, *right_;
};

////////////////////////////////////////////////////////////////////////////////
//
// ENCODER
class Encoder {
public:
    // constructor
    Encoder() : root_(nullptr)
    { for (int i = 0; i < SIZE; ++i) frequency_[i] = sorted_[i] = 0; } // init all array vals to 0

    // methods
    void encode    (const string& in = "text.txt",     // reads txt file and outputs encoded binary file
                    const string& out = "output.bin"); 
    void printFreq () const;                           // prints all freq counts for testing
    void printSort () const;                           // prints sorted_ for testing
    void printDict () const;                           // prints ascii encodings for testing

private:
    // helpers 
    void getFrequencies     (const string& file);                  // reads file and gets freq counts
    void sort               ();                                    // sorts ascii vals by frequency counts
    void buildTree          ();                                    // builds huffman tree
    void buildDict          ();                                    // begin recursive build dict function
    void recursiveBuildDict (Node *root, bool encoding[],          // recursively creates symbol encodings
                             const unsigned int& end); 
    void output             (const string& in, const string& out); // outputs encoded file

    // data
    unsigned int frequency_[SIZE];      // holds frequencies for each ascii value, 
                                        // ascii val corresponds to element index

    unsigned int sorted_[SIZE];         // holds ascii values sorted by frequency, 
                                        // i.e. sorted_[0] = ascii val of least frequent char

    Node         *root_;                // root node of huffman tree
    bool         *dict_[SIZE];          // holds encoding for each ascii val
    unsigned int encodingLength_[SIZE]; // holds the length of the sequence for each ascii val
};

////////////////////////////////////////////////////////////////////////////////
//
// ENCODER functions
////////////////////////////////////////
// reads txt file and outputs encoded binary file
void Encoder::encode(const string& in, const string& out)
{
    getFrequencies(in);
    sort();
    buildTree();
    buildDict();
    output(in, out);
}

////////////////////////////////////////
// reads file and gets freq counts
void Encoder::getFrequencies(const string& file)
{
    // open file
    ifstream input(file);

    // read file and store
    for (unsigned char c; input >> std::noskipws >> c;) 
        ++frequency_[c];

    // close file
    input.close();
}

////////////////////////////////////////
// prints all freq counts for testing
void Encoder::printFreq() const
{
    for (int i = 0; i < SIZE; ++i)
        cout << "ASCII " << i << ": " << frequency_[i] << endl;
}

////////////////////////////////////////
// sorts ascii vals by frequency counts
void Encoder::sort()
{
    // create arrays for device
    unsigned int *frequency_d, *sorted_d;

    // memory sizes to allocate
    int memory = sizeof(int) * SIZE;

    // allocate memory on device
    cudaMalloc((void**)&frequency_d, memory);
    cudaMalloc((void**)&sorted_d, memory);

    // copy frequency and sorted arrays to device
    cudaMemcpy(frequency_d, frequency_, memory, cudaMemcpyHostToDevice);
    cudaMemcpy(sorted_d, sorted_, memory, cudaMemcpyHostToDevice);

    bitonicSort<<<1, SIZE>>>(frequency_d, sorted_d);

    // copy device memory back to host
    cudaMemcpy(sorted_, sorted_d, memory, cudaMemcpyDeviceToHost);
}

////////////////////////////////////////
// prints sorted_ for testing
void Encoder::printSort() const
{
    cout << "Min to Max" << endl;
    for (int i = 0; i < SIZE; ++i)
        cout << "ASCII " << sorted_[i] << ": Frequency " << frequency_[sorted_[i]] << endl;
}

////////////////////////////////////////
// builds huffman tree
void Encoder::buildTree()
{
    // create queues
    queue<Node*> first, second;

    // enqueue all nodes to first queue in increasing order of frequency
    for (int i = 0; i < SIZE; ++i) 
        first.push(new Node(sorted_[i], frequency_[sorted_[i]]));

    // lambda to find min
    auto min = [&]() {
        Node *tmp = nullptr;

        // check if queues are empty
        if (!first.empty() && second.empty())
        {
            tmp = first.front();
            first.pop();
        }
        else if (first.empty() && !second.empty())
        {
            tmp = second.front();
            second.pop();
        }
        // if both queues arent empty
        else if (first.front()->freq_ < second.front()->freq_)
        {
            tmp = first.front();
            first.pop();
        }
        else 
        {
            tmp = second.front();
            second.pop();
        }

        return tmp;
    };

    // begin building
    Node *left, *right, *root;
    while (!(first.empty() && second.size() == 1))
    {
        // get min nodes
        left = min();
        right = min();

        // create new node
        root = new Node(0, left->freq_ + right->freq_); // doesn't matter what the data is for an internal node
        root->left_ = left;
        root->right_ = right;

        // put on second queue
        second.push(root);
    }

    // top root is in the front of the second queue now
    root_ = second.front();
}

////////////////////////////////////////
// begin recursive build dict function
void Encoder::buildDict()
{
    bool encoding[SIZE];

    recursiveBuildDict(root_, encoding, 0);
}

////////////////////////////////////////
// recursively creates symbol encodings
void Encoder::recursiveBuildDict(Node *root, bool encoding[], 
                                 const unsigned int& end)
{
    // if its a leaf node, set encoding
    if (!root->left_ && !root->right_)
    {
        dict_[root->data_] = new bool[end];
        encodingLength_[root->data_] = end;
        for (int i = 0; i < end; ++i)
            dict_[root->data_][i] = encoding[i];

        return;
    }

    // if left node exists, append 0
    if (root->left_)
    {
        encoding[end] = 0;
        recursiveBuildDict(root->left_, encoding, end + 1);
    }
    
    // if right node exists, append 1
    if (root->right_)
    {
        encoding[end] = 1;
        recursiveBuildDict(root->right_, encoding, end + 1);
    }
}

////////////////////////////////////////
// prints ascii encodings for testing
void Encoder::printDict() const
{
    for (int i = 0; i < SIZE; ++i)
    {
        cout << "ASCII " << i << ": ";
        for (int j = 0; j < encodingLength_[i]; ++j)
            cout << dict_[i][j];
        cout << endl;
    }
}

////////////////////////////////////////
// outputs encoded file
void Encoder::output(const string& in, const string& out)
{
    // open file streams
    ifstream input(in);
    ofstream output(out);

    // read input file while outputing encoding
    for (unsigned char c; input >> std::noskipws >> c;)
        for (int i = 0; i < encodingLength_[c]; ++i)
            output << dict_[c][i];
}


////////////////////////////////////////////////////////////////////////////////
//
// KERNEL functions
////////////////////////////////////////
// gpu sort func
__global__ void bitonicSort(unsigned int *freq, unsigned int *sorted)
{
    // cache size of arrays
    const int size = SIZE;

    // create shared arrays
    static __shared__ unsigned int freq_s[size], sorted_s[size];

    // thread idx
    unsigned int idx = threadIdx.x;

    // load 1 elem in each array per index
    freq_s[idx] = freq[idx]; sorted_s[idx] = idx; // ascii vals correspond with array idx

    // bitonic sort alg
    unsigned int tmp, elemIdx1, elemIdx2;
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

            // check if swap is needed
            if ((elemIdx2 < size) && 
                ((max && freq_s[elemIdx1] > freq_s[elemIdx2]) ||
                (!max && freq_s[elemIdx1] < freq_s[elemIdx2])))
            {
                // swap frequencies
                tmp = freq_s[elemIdx1];
                freq_s[elemIdx1] = freq_s[elemIdx2];
                freq_s[elemIdx2] = tmp;

                // swap ascii vals
                tmp = sorted_s[elemIdx1];
                sorted_s[elemIdx1] = sorted_s[elemIdx2];
                sorted_s[elemIdx2] = tmp;
            }     

            // need to sync before next step
            __syncthreads();
        }
    }

    // transfer memory to global
    sorted[idx] = sorted_s[idx];
}

#endif // ENDCODER_H