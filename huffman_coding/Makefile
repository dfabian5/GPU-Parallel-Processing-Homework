main: encoder.cu main.cu
	nvcc -std=c++11 -rdc=true main.cu encoder.cu -o main

clean:
	rm main
	rm output.bin