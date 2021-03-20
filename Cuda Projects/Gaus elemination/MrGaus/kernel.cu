/*****************************************************
*
* Gaussian elimination
*
* Sequential version
*
*****************************************************/
// Compile and then...
// Example run 1:   gauseq.exe -P 1 -I fast -n 16
// Example run 2:   gauseq.exe -P 0 -I rand -n 2048

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>

#define MAX_SIZE 512
// If you put lower NUM_THREADS than BLOCK_SIZE, it will run with BLOCK_SIZE number of threads
#define NUM_THREADS 512
#define BLOCK_SIZE 128

int	N;													/* matrix size				*/
int	maxnum;												/* max number of element	*/
const char *Init;										/* matrix init type			*/
int	PRINT;												/* print switch				*/

double*	A = new double[MAX_SIZE * MAX_SIZE];			/* matrix A					*/
double*	y = new double[MAX_SIZE];						/* vector y					*/

/* forward declarations */
void work_gpu(void);
void Init_Matrix(void);
void Print_Matrix(void);
void Init_Default(void);
void Init_Test(void);
int Read_Options(int, char **);
int matrixAt(int x, int y);

__device__ int d_matrixAt(int x, int y)
{
	return x * MAX_SIZE + y;
}

__global__ void GausElimination(double A[MAX_SIZE * MAX_SIZE], double y[MAX_SIZE], int k, int threads)
{
	// Shared memory for the k:th row
	__shared__ double k_row[MAX_SIZE];

	int thread = threadIdx.x;
	// A whole block load in shared memory
	while (thread < MAX_SIZE)
	{
		k_row[thread] = A[d_matrixAt(thread, k)];
		thread += BLOCK_SIZE;
	}

	// Wait for all threads to allocate shared memory
	__syncthreads();

	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int j = idx + k + 1;
	double p = k_row[k];

	// When k increases, we need to use less threads.
	while (j < MAX_SIZE)
	{
		// factor for a row to divide with
		double f = A[d_matrixAt(k, j)] / p;

		for (int i = k; i < MAX_SIZE; i++)
		{
			A[d_matrixAt(i, j)] = A[d_matrixAt(i, j)] - k_row[i] * f;
		}

		y[j] = y[j] - y[k] * f;

		j += threads;
	}
}

__global__ void GausDivide(double A[MAX_SIZE * MAX_SIZE], double y[MAX_SIZE], int k, int threads)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int i = idx + k;

	double p = A[d_matrixAt(k, k)];

	
	// Let one thread divide the y value
	if (idx == 0)
	{
		y[k] = y[k] / p;
	}
	
	// thread coarsning
	while (i < MAX_SIZE)
	{
		// Divide the k row with its pivot
		A[d_matrixAt(i, k)] = A[d_matrixAt(i, k)] / p;

		i += threads;
	}
}

void GausBack()
{
	// Backpropegate to get y
	for (int k = MAX_SIZE - 1; k >= 0; k--)
	{
		for (int j = k - 1; j >= 0; j--)
		{
			y[j] = y[j] - y[k] * A[matrixAt(k, j)];
			A[matrixAt(k, j)] = 0.0;
		}
	}
}

int matrixAt(int x, int y)
{
	return x * MAX_SIZE + y; 
}

void work_cpu()
{
	for (int k = 0; k < MAX_SIZE; k++)
	{
		// Save the pivot value
		float p = A[matrixAt(k, k)];

		//  (i, j) is (x,y)
		for (int j = k + 1; j < MAX_SIZE; j++)
		{
			// factor for a row to divide with
			float f = A[matrixAt(k, j)] / p;

			for (int i = k; i < MAX_SIZE; i++)
			{
				A[matrixAt(i, j)] = A[matrixAt(i, j)] - A[matrixAt(i, k)] * f;			
			}
			y[j] = y[j] - y[k] * f;		
		}

		// Divide the k row with its pivot
		for (int i = k; i < MAX_SIZE; i++)
		{
			A[matrixAt(i, k)] = A[matrixAt(i, k)] / p;
		}
		y[k] = y[k] / p;
	}
}

void work_gpu(void)
{
	double* d_A;
	double* d_y;

	int sizeY = sizeof(double) * MAX_SIZE;
	int sizeA = sizeY * MAX_SIZE;

	cudaMalloc((void**)&d_A, sizeA);
	cudaMalloc((void**)&d_y, sizeY);

	cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, sizeY, cudaMemcpyHostToDevice);

	int threads = ceil((float)NUM_THREADS / (float)BLOCK_SIZE) * BLOCK_SIZE;

	// Gaussian elimination algorithm
	for (int k = 0; k < MAX_SIZE; k++) // Outer loop
	{
		// Max number of blocks is 65535 so our application can't handle N > 65536
		GausElimination << <ceil((float)NUM_THREADS / (float)BLOCK_SIZE), BLOCK_SIZE >> > (d_A, d_y, k, threads);
		GausDivide << < ceil((float)NUM_THREADS / (float)BLOCK_SIZE), BLOCK_SIZE >> > (d_A, d_y, k, threads);
	}


	cudaMemcpy(A, d_A, sizeA, cudaMemcpyDeviceToHost);
	cudaMemcpy(y, d_y, sizeY, cudaMemcpyDeviceToHost);
}

int main(int argc, char **argv)
{
	int i, timestart, timeend, iter;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	Init_Default();				/* Init default values	*/
	Read_Options(argc, argv);	/* Read arguments	*/
	Init_Matrix();				/* Init the matrix	*/

	//work_cpu();
	work_gpu();
	// For fun, not really necessary but we get the identity matrix
	//GausBack();
	

	

	/*
	printf("Y: [");
	for (int i = 0; i < N; i++)
	{
		printf("%5.2f, ", y[i]);
	}
	printf("]");
	*/

	if (PRINT == 1) Print_Matrix();
}

void Init_Matrix()
{
	int i, j;

	/*
	printf("\nsize      = %dx%d ", N, N);
	printf("\nmaxnum    = %d \n", maxnum);
	printf("Init	  = %s \n", Init);
	printf("Initializing matrix...");
	*/

	if (strcmp(Init, "rand") == 0)
	{
		for (i = 0; i < N; i++)
		{
			for (j = 0; j < N; j++)
			{
				if (i == j) /* diagonal dominance */
					A[matrixAt(j, i)] = (double)(rand() % maxnum) + 5.0;
				else
					A[matrixAt(j, i)] = (double)(rand() % maxnum) + 1.0;
			}
		}
	}
	if (strcmp(Init, "fast") == 0)
	{
		for (i = 0; i < N; i++)
		{
			for (j = 0; j < N; j++)
			{
				if (i == j) /* diagonal dominance */
					A[matrixAt(j, i)] = 5.0;
				else
					A[matrixAt(j ,i)] = 2.0;
			}
		}
	}

	/* Initialize vectors b and y */
	for (i = 0; i < N; i++)
	{
		y[i] = 2.0;
	}

	//printf("done \n\n");
	if (PRINT == 1)
		Print_Matrix();
}

void Print_Matrix()
{
	int i, j;

	
	bool printA = false;

	if (printA)
	{
		printf("Matrix A:\n");
		for (i = 0; i < N; i++)
		{
			printf("[");
			for (j = 0; j < N; j++)
				printf(" %5.2f,", A[matrixAt(j, i)]);
			printf("]\n");
		}
	}
	

	printf("Vector y:\n[");
	for (j = 0; j < N; j++)
		printf(" %5.2f,", y[j]);
	printf("]\n");
	printf("\n\n");
}

void Init_Default()
{
	N = MAX_SIZE;
	Init = "rand";
	maxnum = 15.0;
	PRINT = 0;
}


int Read_Options(int argc, char **argv)
{
	char    *prog;

	prog = *argv;
	while (++argv, --argc > 0)
		if (**argv == '-')
			switch (*++*argv)
			{
			case 'n':
				--argc;
				N = atoi(*++argv);
				break;
			case 'h':
				printf("\nHELP: try sor -u \n\n");
				exit(0);
				break;
			case 'u':
				printf("\nUsage: sor [-n problemsize]\n");
				printf("           [-D] show default values \n");
				printf("           [-h] help \n");
				printf("           [-I init_type] fast/rand \n");
				printf("           [-m maxnum] max random no \n");
				printf("           [-P print_switch] 0/1 \n");
				exit(0);
				break;
			case 'D':
				printf("\nDefault:  n         = %d ", N);
				printf("\n          Init      = rand");
				printf("\n          maxnum    = 5 ");
				printf("\n          P         = 0 \n\n");
				exit(0);
				break;
			case 'I':
				--argc;
				Init = *++argv;
				break;
			case 'm':
				--argc;
				maxnum = atoi(*++argv);
				break;
			case 'P':
				--argc;
				PRINT = atoi(*++argv);
				break;
			default:
				printf("%s: ignored option: -%s\n", prog, *argv);
				printf("HELP: try %s -u \n\n", prog);
				break;
			}
}
