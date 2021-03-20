
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "CImg.h"
#include <vector>
#include <iostream>
#include <chrono>

using namespace cimg_library;

#define BLOCK_SIZEX 16
#define BLOCK_SIZEY 16

// Biggest picture will be (65535*BLOCK_SIZEX, 65535*BLOCK_SIZEY)
// If we increase BLOCK_SIZE we can get even bigger pictures. (Block_size = 32)

struct rgb
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

int matrixAt(int x, int y, unsigned int width)
{
    return x + y * width;
}

__device__ int matrixAt_d(int x, int y, unsigned int width)
{
    return x + y * width;
}

__global__ void dataTransformation(rgb* AoS_d, unsigned char* SoA_d, unsigned int width, unsigned int height, unsigned int color)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    // Don't go out of memory
    if (idx < width && idy < height)
    {
        // Causes no thread divergence since it is only called with a constant color 
        if (color == 0)
        {
            SoA_d[matrixAt_d(idx, idy, width)] = AoS_d[matrixAt_d(idx, idy, width)].r;
        }
        else if (color == 1)
        {
            SoA_d[matrixAt_d(idx, idy, width)] = AoS_d[matrixAt_d(idx, idy, width)].g;
        }
        else if (color == 2)
        {
            SoA_d[matrixAt_d(idx, idy, width)] = AoS_d[matrixAt_d(idx, idy, width)].b;
        }
    }
}

__global__ void ImageBlur(unsigned char* SoA_d, unsigned char* SoA_blur, double* mask_d, unsigned int width, unsigned int height)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    double tempSum = 0;

    // Do not go outside the memory
    if (idx < width && idy < height)
    {
        for (int x = -2; x <= 2; x++)
        {
            for (int y = -2; y <= 2; y++)
            {
                // Do not sum values from outside the picture
                if (idx + x >= 0 && idx + x < width && idy + y >= 0 && idy + y < height)
                {
                    tempSum += mask_d[matrixAt_d(x + 2, y + 2, 5)] * (double)SoA_d[matrixAt_d(idx + x, idy + y, width)];
                }
            }
        }
    }

    // Clamp the value to 0-255
    if (tempSum > 255) tempSum = 255;
    else if (tempSum < 0) tempSum = 0;

    SoA_blur[matrixAt_d(idx, idy, width)] = (char)tempSum;
}

void showImage(unsigned char* SoA[3], CImg<unsigned char>& image)
{
    // For showing the picture with Cimg
    for (int x = 0; x < image._width; x++)
    {
        for (int y = 0; y < image._height; y++)
        {
            image(x, y, 0, 0) = SoA[0][matrixAt(x, y, image._width)];
            image(x, y, 0, 1) = SoA[1][matrixAt(x, y, image._width)];
            image(x, y, 0, 2) = SoA[2][matrixAt(x, y, image._width)];
        }
    }

    CImgDisplay main_disp(image, "GPU - Blurred image");

    while (1)
    {
        main_disp.wait();
    }
}


int main()
{
    CImg<unsigned char> image("cake.ppm"), image_cpu("cake.ppm");

    const int size = image._width * image._height;

    // Array of Struct
    rgb* rgbStruct;
    rgbStruct = new rgb[size];

    // "Struct of Array"
    unsigned char* SoA[3];
    SoA[0] = new unsigned char[size];
    SoA[1] = new unsigned char[size];
    SoA[2] = new unsigned char[size];

    for (int x = 0; x < image._width; x++)
    {
        for (int y = 0; y < image._height; y++)
        {
            // Put the rgb values in to the rgb struct array
            rgbStruct[matrixAt(x, y, image._width)].r = image(x, y, 0, 0);
            rgbStruct[matrixAt(x, y, image._width)].g = image(x, y, 0, 1);
            rgbStruct[matrixAt(x, y, image._width)].b = image(x, y, 0, 2);
        }
    }

    // Declare device variables
    rgb* AoS_d = nullptr;
    unsigned char* SoA_d[3];
    SoA_d[0] = nullptr;
    SoA_d[1] = nullptr;
    SoA_d[2] = nullptr;

    // Allocate memory on device
    cudaMalloc((void**)&AoS_d, sizeof(rgb)*size);
    cudaMalloc((void**)&SoA_d[0], size);
    cudaMalloc((void**)&SoA_d[1], size);
    cudaMalloc((void**)&SoA_d[2], size);

    // Send over the Array of Structure
    cudaMemcpy(AoS_d, rgbStruct, size*sizeof(rgb), cudaMemcpyHostToDevice);

    // Create a grid with correct amount of threads and blocks
    dim3 numBlocks(ceil((float)image._width / (float)BLOCK_SIZEX), ceil((float)image._height / (float)BLOCK_SIZEY));
    dim3 blockSize(BLOCK_SIZEX, BLOCK_SIZEY);

    // Kernel call to swap array of structure to structure of arrays
    dataTransformation << <numBlocks, blockSize >> > (AoS_d, SoA_d[0], image._width, image._height, 0); // R
    dataTransformation << <numBlocks, blockSize >> > (AoS_d, SoA_d[1], image._width, image._height, 1); // G
    dataTransformation << <numBlocks, blockSize >> > (AoS_d, SoA_d[2], image._width, image._height, 2); // B

    // Send back the result to CPU
    cudaMemcpy(SoA[0], SoA_d[0], size, cudaMemcpyDeviceToHost);
    cudaMemcpy(SoA[1], SoA_d[1], size, cudaMemcpyDeviceToHost);
    cudaMemcpy(SoA[2], SoA_d[2], size, cudaMemcpyDeviceToHost);

    // Variable for blurred channel
    unsigned char* SoA_blur = nullptr;

    // Set up the mask
    double* mask_d = nullptr;
    double mask[5*5];

    mask[matrixAt(0, 1, 5)] = mask[matrixAt(0, 3, 5)] = mask[matrixAt(1, 0, 5)] = mask[matrixAt(1, 4, 5)] =
    mask[matrixAt(3, 0, 5)] = mask[matrixAt(3, 4, 5)] = mask[matrixAt(4, 1, 5)] = mask[matrixAt(4, 3, 5)] = 4.0 / 256.0;
    mask[matrixAt(0, 0, 5)] = mask[matrixAt(0, 4, 5)] = mask[matrixAt(4, 0, 5)] = mask[matrixAt(4, 4, 5)] = 1.0 / 256.0;
    mask[matrixAt(0, 2, 5)] = mask[matrixAt(2, 0, 5)] = mask[matrixAt(2, 4, 5)] = mask[matrixAt(4, 2, 5)] = 6.0 / 256.0;
    mask[matrixAt(1, 1, 5)] = mask[matrixAt(1, 3, 5)] = mask[matrixAt(3, 1, 5)] = mask[matrixAt(3, 3, 5)] = 16.0 / 256.0;
    mask[matrixAt(1, 2, 5)] = mask[matrixAt(2, 1, 5)] = mask[matrixAt(2, 3, 5)] = mask[matrixAt(3, 2, 5)] = 24.0 / 256.0;
    mask[matrixAt(2, 2, 5)] = 36.0 / 256.0;

    // Allocate memory
    cudaMalloc((void**)&SoA_blur, size);
    cudaMalloc((void**)&mask_d, sizeof(double) * 5 * 5);
    cudaMemcpy(mask_d, mask, sizeof(double) * 5 * 5, cudaMemcpyHostToDevice);

    // Kernel call to gauss blur for each channel
    ImageBlur << <numBlocks, blockSize >> > (SoA_d[0], SoA_blur, mask_d, image._width, image._height); // R
    cudaMemcpy(SoA[0], SoA_blur, size, cudaMemcpyDeviceToHost);

    ImageBlur << <numBlocks, blockSize >> > (SoA_d[1], SoA_blur, mask_d, image._width, image._height); // G
    cudaMemcpy(SoA[1], SoA_blur, size, cudaMemcpyDeviceToHost);

    ImageBlur << <numBlocks, blockSize >> > (SoA_d[2], SoA_blur, mask_d, image._width, image._height); // B
    cudaMemcpy(SoA[2], SoA_blur, size, cudaMemcpyDeviceToHost);

    showImage(SoA, image);

    /*
    // Test
    for (int x = 0; x < image._width; x++)
    {
        for (int y = 0; y < image._height; y++)
        {
            if (SoA[0][matrixAt(x, y, image._width)] != image(x, y, 0, 0) &&
                SoA[1][matrixAt(x, y, image._width)] != image(x, y, 0, 1) &&
                SoA[2][matrixAt(x, y, image._width)] != image(x, y, 0, 2))
                printf("ErroR: @ (%d,%d) :: (%d, %d, %d)\n", x, y, SoA[0][matrixAt(x, y, image._width)], SoA[1][matrixAt(x, y, image._width)], SoA[2][matrixAt(x, y, image._width)]);
        }
    }
    */

    return 0;
}
