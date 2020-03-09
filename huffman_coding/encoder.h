#ifndef ENDCODER_H
#define ENDCODER_H

////////////////////////////////////////////////////////////////////////////////
//
// FILE:        encoder.h
// DESCRIPTION: contains class for huffman coding a text file
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

#endif // ENDCODER_H