// MP 4 -- 2D Convolution
#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2

//@@ INSERT CODE HERE

#define BLOCK_WIDTH   16
#define BLOCK_HEIGHT  16

#define MASK_WIDTH 5
#define MASK_HEIGHT 5

#define O_TILE_WIDTH  (BLOCK_WIDTH - (MASK_WIDTH-1))
#define O_TILE_HEIGHT (BLOCK_HEIGHT - (MASK_HEIGHT-1))

__global__ void convolution_2D_kernel(float *inputData, float *outputData, int height, int width, int channels, const float* __restrict__ maskData)
{
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	
    int outputRow = blockIdx.y*O_TILE_HEIGHT + ty;
    int outputCol = blockIdx.x*O_TILE_WIDTH  + tx;

    int inputRow = outputRow - MASK_HEIGHT/2;
    int inputCol = outputCol - MASK_WIDTH/2;

    __shared__ float inputTile[BLOCK_HEIGHT][BLOCK_WIDTH];

    for(int channel=0;channel<channels;channel++)
    {
        __syncthreads();

        if (inputRow < 0 || inputRow >=height || inputCol < 0 || inputCol >= width)
        {
            inputTile[ty][tx] = 0.0;
        }
        else
        {
            inputTile[ty][tx] = inputData[(inputRow*width + inputCol)*channels + channel];
        }
        __syncthreads();


        if (ty < O_TILE_HEIGHT && tx < O_TILE_WIDTH && outputRow < height && outputCol < width)
        {
            float output = 0.0; 

            for(int y=0;y<MASK_HEIGHT;y++)
            {
                for(int x=0;x<MASK_WIDTH;x++)
                {
                    output += maskData[y*MASK_WIDTH + x] * inputTile[y + ty][x + tx];
                }
            }
            outputData[(outputRow*width + outputCol)*channels + channel] = output;
        }
    }
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");

    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE

    dim3 dimGrid(1+(imageWidth-1)/O_TILE_WIDTH, 1+(imageHeight-1)/O_TILE_HEIGHT, 1);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    printf("block=(%d,%d,%d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
    printf("grid=(%d,%d,%d)\n", dimGrid.x, dimGrid.y, dimGrid.z);
    printf("height=%d, width=%d, channels=%d\n", imageHeight, imageWidth, imageChannels);
    printf("O_TILE_WIDTH=%d O_TILE_HEIGHT=%d\n", O_TILE_WIDTH, O_TILE_HEIGHT);

    wbTime_stop(Compute, "Doing the computation on the GPU");
    convolution_2D_kernel<<<dimGrid,dimBlock>>>(deviceInputImageData, deviceOutputImageData, imageHeight, imageWidth, imageChannels, deviceMaskData);
    cudaThreadSynchronize();

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

