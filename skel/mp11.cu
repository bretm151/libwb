// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256

const float WEIGHT_R = 0.21f;
const float WEIGHT_G = 0.71f;
const float WEIGHT_B = 0.07f;

typedef unsigned char UCHAR, *PUCHAR;
typedef unsigned int UINT, *PUINT;

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", err);                        \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return err;                                                       \
        }                                                                     \
    } while(0)

//@@ insert code here

cudaError_t
hostCastToUChar(const float *floatImage, PUCHAR ucharImage, int height, int width, int channels)
{
    for(int i=0;i<height*width*channels;i++)
    {
        ucharImage[i] = (unsigned char)(255.0f * floatImage[i]);
    }

    return cudaSuccess;
}

__global__
void gpu_castToUChar(const float *floatImage, PUCHAR ucharImage, int height, int width, int channels)
{
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
    const int idx = (blockIdx.y*blockDim.y + ty)*width + blockIdx.x*blockDim.x + tx;

    if (idx < height*width)
    {
        for(int i=0;i<channels;i++)
        {
            int index = idx*3 + i;
            ucharImage[index] = (UCHAR)(255.0f * floatImage[index]);
        }
    }
}

cudaError_t
deviceCastToUChar(const float *floatImage, PUCHAR ucharImage, int height, int width, int channels)
{
    int numElements = height*width*channels;
    dim3 dimBlock(31, 31, 1);
    dim3 dimGrid(1+(width-1)/dimBlock.x, 1+(height-1)/dimBlock.y, 1);

    float *gpu_floatImage=NULL;
    PUCHAR gpu_ucharImage=NULL;

    wbCheck(cudaMalloc((void**)&gpu_floatImage, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&gpu_ucharImage, numElements*sizeof(UCHAR)));

    wbCheck(cudaMemcpy(gpu_floatImage, floatImage, numElements*sizeof(float), cudaMemcpyHostToDevice));

    gpu_castToUChar<<<dimGrid, dimBlock>>>(gpu_floatImage, gpu_ucharImage, height, width, channels);

    wbCheck(cudaThreadSynchronize());

    wbCheck(cudaMemcpy(ucharImage, gpu_ucharImage, numElements*sizeof(UCHAR), cudaMemcpyDeviceToHost));
    wbCheck(cudaFree(gpu_floatImage));
    wbCheck(cudaFree(gpu_ucharImage));

    return cudaSuccess;
}

cudaError_t
castToUChar(const float *floatImage, PUCHAR ucharImage, int height, int width, int channels)
{
    PUCHAR host_ucharImage=(PUCHAR)malloc(height*width*channels*sizeof(UCHAR));

    hostCastToUChar(floatImage, host_ucharImage, height, width, channels);
    deviceCastToUChar(floatImage, ucharImage, height, width, channels);

    for(int i=0;i<height*width*channels;i++)
    {
        if(host_ucharImage[i] != ucharImage[i])
        {
            wbLog(ERROR, "miscompare at ", i);                       \
        }
    }

    free(host_ucharImage);

    return cudaSuccess;
}

cudaError_t
convertRGBToGray(const PUCHAR rgbImage, PUCHAR grayImage, int height, int width, int channels)
{
    assert(channels == 3);

    for(int y=0;y<height;y++)
    {
        for(int x=0;x<width;x++)
        {
            int idx = y*width + x;

            float r = rgbImage[channels*idx + 0];
            float g = rgbImage[channels*idx + 1];
            float b = rgbImage[channels*idx + 2];

            grayImage[idx] = (UCHAR)(r*WEIGHT_R + g*WEIGHT_G + b*WEIGHT_B);
        }
    }

    return cudaSuccess;
}

cudaError_t
computeHistogram(const PUCHAR grayImage, PUINT histogram, int height, int width)
{
    for(int i=0;i<HISTOGRAM_LENGTH;i++)
    {
        histogram[i] = 0;
    }

    for(int i=0;i<height*width;i++)
    {
        histogram[grayImage[i]]++;
    }

    return cudaSuccess;
}

cudaError_t
computeCDF(const PUINT histogram, float *cdf, int height, int width)
{
    float prev = 0.0;

    for(int i=0;i<HISTOGRAM_LENGTH;i++)
    {
        cdf[i] = prev + histogram[i]/(float)(height*width);
        prev = cdf[i];
    }

    return cudaSuccess;
}

UCHAR
clamp(const UCHAR x, const UCHAR start, const UCHAR end)
{
    return min(max(x,start), end);
}

UCHAR
correctColor(UCHAR val, float *cdf)
{
    return clamp((UCHAR)(255*(cdf[val]-cdf[0])/(1-cdf[0])), 0, 255);
}

cudaError_t
correctColors(PUCHAR ucharImage, float *cdf, int height, int width, int channels)
{
    for(int i=0;i<height*width*channels;i++)
    {
        ucharImage[i] = correctColor(ucharImage[i], cdf);
    }

    return cudaSuccess;
}

cudaError_t
castToFloat(const PUCHAR input, float *output, int height, int width, int channels)
{
    for(int i=0;i<height*width*channels;i++)
    {
        output[i] = (float)(input[i]/255.0f);
    }

    return cudaSuccess;
}

cudaError_t
equalizeHistogram(const float *inputData, float * outputData, int height, int width, int channels)
{
    PUCHAR ucharImage = NULL;
    PUCHAR grayImage  = NULL;
    PUINT  histogram  = NULL;
    float *cdf        = NULL;

    wbCheck(cudaMallocHost((void **)&ucharImage, sizeof(UCHAR) * height * width * channels, cudaHostAllocDefault));
    wbCheck(cudaMallocHost((void **)&grayImage, sizeof(UCHAR) * height * width, cudaHostAllocDefault));
    wbCheck(cudaMallocHost((void **)&histogram, sizeof(UINT) * HISTOGRAM_LENGTH, cudaHostAllocDefault));
    wbCheck(cudaMallocHost((void **)&cdf, sizeof(float) * HISTOGRAM_LENGTH, cudaHostAllocDefault));

    wbCheck(castToUChar(inputData, ucharImage, height, width, channels));
    wbCheck(convertRGBToGray(ucharImage, grayImage, height, width, channels));
    wbCheck(computeHistogram(grayImage, histogram, height, width));
    wbCheck(computeCDF(histogram, cdf, height, width));
    wbCheck(correctColors(ucharImage, cdf, height, width, channels));
    wbCheck(castToFloat(ucharImage, outputData, height, width, channels));

    cudaFreeHost(cdf);
    cudaFreeHost(histogram);
    cudaFreeHost(grayImage);
    cudaFreeHost(ucharImage);

    return cudaSuccess;
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    const char * inputImageFile;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here
    wbLog(TRACE, "The image height is ", imageHeight);
    wbLog(TRACE, "The image width is ",  imageWidth);
    wbLog(TRACE, "The number of channels is ",  imageChannels);
    wbLog(TRACE, "The number of pixels is ",  imageHeight*imageWidth*imageChannels);

    wbCheck(equalizeHistogram(wbImage_getData(inputImage),
                              wbImage_getData(outputImage),
                              imageHeight,
                              imageWidth,
                              imageChannels));

    wbSolution(args, outputImage);

    //@@ insert code here
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

