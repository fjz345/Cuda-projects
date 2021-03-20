#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <iostream>
#include <fstream>
#include <cstring>
#include <random>

#define ARRAYSIZE (100000)
#define NUM_THREADS (1024) // Always 1 more thread if arraysize is odd
#define BLOCK_SIZE 128

using namespace std;

void PrintText(int out[ARRAYSIZE], int out2[ARRAYSIZE])
{
	for (int i = 0; i < ARRAYSIZE; i++)
	{
		printf("[%d]: %d \"%d\" \n", i, out2[i], out2[i] - out[i]);
	}
}

int * OddEvenSortSingleThread(int in[])
{
	int *out = new int[ARRAYSIZE];

	for (int i = 0; i < ARRAYSIZE; i++)
	{
		out[i] = in[i];
	}

	// Run the sorting ARRAYSIZE times to make sure that everything is sorted
	// NOTE: We have two steps at once
	// However, if the ARRAYSIZE is an odd number, we need to make sure that the loop runs enough times 
	for (int i = 0; i < (ARRAYSIZE + 1) / 2; i++)
	{
		// Even
		for (int j = 0; j < ARRAYSIZE - 1; j += 2)
		{
			if (out[j] > out[j + 1])
			{
				swap(out[j], out[j + 1]);
			}
		}

		// Odd
		for (int j = 1; j < ARRAYSIZE - 1; j += 2)
		{
			if (out[j] > out[j + 1])
			{
				swap(out[j], out[j + 1]);
			}
		}
	}

	return out;
}

__device__ void swap_d(int randomNum_d[], int id, int threadStep)
{
	int temp;
	int thisID = id;
	int nextID = id + 1;

	int step = 0;
	while (nextID + step < ARRAYSIZE)
	{
		if (randomNum_d[thisID + step] > randomNum_d[nextID + step])
		{
			temp = randomNum_d[thisID + step];
			randomNum_d[thisID + step] = randomNum_d[nextID + step];
			randomNum_d[nextID + step] = temp;
		}

		step += threadStep;
	}
}

__global__ void EvenSortParallel(int randomNum_d[], int threadStep)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	// EVEN
	// Swap values the thread is assigned with
	swap_d(randomNum_d, id * 2, threadStep);
	__syncthreads();
}

__global__ void OddSortParallel(int randomNum_d[], int threadStep)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	// ODD
	// Swap values the thread is assigned with
	swap_d(randomNum_d, id * 2 + 1, threadStep);
	__syncthreads();
}


int main(int argc, char* argv[])
{
	bool saveToFile = true;
	// Loop through args and check for -nodebug
	for (int i = 0; i < argc; ++i)
	{
		if (std::string(argv[i]) == "-noFile") saveToFile = false;
	}

	/* initialize random seed: */
	srand(time(NULL));

	std::random_device rd;
	std::uniform_int_distribution<int> dist;
	
	int* randomNum = new int[ARRAYSIZE];
	for (int i = 0; i < ARRAYSIZE; i++)
	{
		randomNum[i] = dist(rd) % ARRAYSIZE;
	}

	// Parallel --------------------------------------------
	
	int size = ARRAYSIZE * sizeof(int);
	int *out2 = new int[ARRAYSIZE];
	int threadStep = ceil((float) NUM_THREADS / (float) BLOCK_SIZE) * BLOCK_SIZE * 2;

	int *randomNum_d;

	// Transfer array to device memory
	cudaMalloc((void**)&randomNum_d, size);

	cudaMemcpy(randomNum_d, randomNum, size, cudaMemcpyHostToDevice);

	for (int i = 0; i < (ARRAYSIZE + 1) / 2; i++)
	{
		// Run ceil(ARRAYSIZE/NUM_THREADS) block of NUM_THREADS threads each --- function<<<grid, threads>>>();
		EvenSortParallel <<< ceil((float)NUM_THREADS / (float)BLOCK_SIZE), BLOCK_SIZE >>> (randomNum_d, threadStep);

		OddSortParallel <<< ceil((float)NUM_THREADS / (float)BLOCK_SIZE), BLOCK_SIZE >>> (randomNum_d, threadStep);

	}

	// Transfer array from device to host
	cudaMemcpy(out2, randomNum_d, size, cudaMemcpyDeviceToHost);
	

	// Single Thread --------------------------------------

	/*
	int * out = new int[ARRAYSIZE];
	out = OddEvenSortSingleThread(randomNum);
	*/

	// Debug the output
	//PrintText(out, out2);

	
	// Save result to file
	if (saveToFile)
	{
		std::string fileName = "gpuSorted.txt";
		ofstream outputFile(fileName);

		if (outputFile.is_open())
		{
			for (int i = 0; i < ARRAYSIZE; i++)
			{
				outputFile << out2[i] << std::endl;
			}

			outputFile.close();
		}
		else
		{
			std::cout << fileName.c_str() << " could not be opened" << std::endl;
		}
	}

	

	// Free memory
	cudaFree(randomNum_d);

	return 0;
}
