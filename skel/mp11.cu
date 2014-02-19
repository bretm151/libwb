// Histogram Equalization

#include    <wb.h>
#include <cuda.h>

#define HISTOGRAM_LENGTH 256

//#define RUN_ON_HOST

#define WEIGHT_R   0.21f
#define WEIGHT_G   0.71f
#define WEIGHT_B   0.07f

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

#if defined(RUN_ON_HOST)

cudaError_t
castToUChar(const float *floatImage, PUCHAR ucharImage, int height, int width, int channels)
{
    for(int i=0;i<height*width*channels;i++)
    {
        ucharImage[i] = (unsigned char)(255.0f * floatImage[i]);
    }

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
            int index = y*width + x;

            float r = rgbImage[channels*index + 0];
            float g = rgbImage[channels*index + 1];
            float b = rgbImage[channels*index + 2];

            grayImage[index] = (UCHAR)(r*WEIGHT_R + g*WEIGHT_G + b*WEIGHT_B);
        }
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

#else

__global__
void gpu_castToUChar(const float *floatImage, PUCHAR ucharImage, int height, int width, int channels)
{
	const int ty = threadIdx.y;
	const int tx = threadIdx.x;
    const int idx = (blockIdx.y + ty)*width + blockIdx.x*blockDim.x + tx;

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
castToUChar(const float *floatImage, PUCHAR ucharImage, int height, int width, int channels)
{
    dim3 dimBlock(1, 1, 1);
    dim3 dimGrid(1+(width-1)/dimBlock.x, 1+(height-1)/dimBlock.y, 1);

    float *gpu_floatImage=NULL;
    PUCHAR gpu_ucharImage=NULL;

    wbCheck(cudaHostGetDevicePointer((void **)&gpu_floatImage, (void *)floatImage, 0));
    wbCheck(cudaHostGetDevicePointer((void **)&gpu_ucharImage, (void *)ucharImage, 0));

    gpu_castToUChar<<<dimGrid, dimBlock>>>(gpu_floatImage, gpu_ucharImage, height, width, channels);

    wbCheck(cudaThreadSynchronize());

    return cudaSuccess;
}

__global__
void gpu_convertRGBToGray(const PUCHAR rgbImage, PUCHAR grayImage, int height, int width, int channels)
{
	const int ty = threadIdx.y;
	const int tx = threadIdx.x;
    const int idx = (blockIdx.y + ty)*width + blockIdx.x*blockDim.x + tx;

    if (idx < height*width)
    {
        int index = idx*3;

        float r = rgbImage[channels*index + 0];
        float g = rgbImage[channels*index + 1];
        float b = rgbImage[channels*index + 2];

        grayImage[idx] = (UCHAR)(r*WEIGHT_R + g*WEIGHT_G + b*WEIGHT_B);
    }
}

cudaError_t
convertRGBToGray(const PUCHAR rgbImage, PUCHAR grayImage, int height, int width, int channels)
{
    dim3 dimBlock(1, 1, 1);
    dim3 dimGrid(1+(width-1)/dimBlock.x, 1+(height-1)/dimBlock.y, 1);

    PUCHAR gpu_rgbImage=NULL;
    PUCHAR gpu_grayImage=NULL;

    assert(channels == 3);

    wbCheck(cudaHostGetDevicePointer((void **)&gpu_rgbImage, (void *)rgbImage, 0));
    wbCheck(cudaHostGetDevicePointer((void **)&gpu_grayImage, (void *)grayImage, 0));

    gpu_convertRGBToGray<<<dimGrid, dimBlock>>>(gpu_rgbImage, gpu_grayImage, height, width, channels);

    wbCheck(cudaThreadSynchronize());

    return cudaSuccess;
}

__device__
UCHAR
clamp(const UCHAR x, const UCHAR start, const UCHAR end)
{
    return min(max(x,start), end);
}

__device__
UCHAR
correctColor(UCHAR val, float *cdf)
{
    return clamp((UCHAR)(255*(cdf[val]-cdf[0])/(1-cdf[0])), 0, 255);
}

__global__
void gpu_correctColors(PUCHAR ucharImage, float *cdf, int height, int width, int channels)
{
	const int ty = threadIdx.y;
	const int tx = threadIdx.x;
    const int idx = (blockIdx.y + ty)*width + blockIdx.x*blockDim.x + tx;

    if (idx < height*width)
    {
        for(int i=0;i<channels;i++)
        {
            int index = idx*3 + i;
            ucharImage[index] = correctColor(ucharImage[index], cdf);
        }
    }
}

cudaError_t
correctColors(PUCHAR ucharImage, float *cdf, int height, int width, int channels)
{
    dim3 dimBlock(1, 1, 1);
    dim3 dimGrid(1+(width-1)/dimBlock.x, 1+(height-1)/dimBlock.y, 1);

    PUCHAR gpu_ucharImage=NULL;
    float *gpu_cdf=NULL;

    wbCheck(cudaHostGetDevicePointer((void **)&gpu_ucharImage, (void *)ucharImage, 0));
    wbCheck(cudaHostGetDevicePointer((void **)&gpu_cdf, (void *)cdf, 0));

    gpu_correctColors<<<dimGrid, dimBlock>>>(gpu_ucharImage, gpu_cdf, height, width, channels);

    wbCheck(cudaThreadSynchronize());

    return cudaSuccess;
}

__global__
void gpu_castToFloat(const PUCHAR ucharImage, float *floatImage, int height, int width, int channels)
{
	const int ty = threadIdx.y;
	const int tx = threadIdx.x;
    const int idx = (blockIdx.y + ty)*width + blockIdx.x*blockDim.x + tx;

    if (idx < height*width)
    {
        for(int i=0;i<channels;i++)
        {
            int index = idx*3 + i;
            floatImage[index] = (float)(ucharImage[index]/255.0f);
        }
    }
}

cudaError_t
castToFloat(const PUCHAR ucharImage, float *floatImage, int height, int width, int channels)
{
    dim3 dimBlock(1, 1, 1);
    dim3 dimGrid(1+(width-1)/dimBlock.x, 1+(height-1)/dimBlock.y, 1);

    PUCHAR gpu_ucharImage=NULL;
    float *gpu_floatImage=NULL;

    wbCheck(cudaHostGetDevicePointer((void **)&gpu_ucharImage, (void *)ucharImage, 0));
    wbCheck(cudaHostGetDevicePointer((void **)&gpu_floatImage, (void *)floatImage, 0));

    gpu_castToFloat<<<dimGrid, dimBlock>>>(gpu_ucharImage, gpu_floatImage, height, width, channels);

    wbCheck(cudaThreadSynchronize());

    return cudaSuccess;
}

#endif

#if defined(RUN_ON_HOST)
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

#else

__global__
void gpu_computeHistogram(const PUCHAR grayImage, PUINT histogram, int height, int width)
{
	const int tx = threadIdx.x;
    const int idx = blockIdx.x*blockDim.x + tx;
    const int stride = blockDim.x * gridDim.x;

    __shared__ UINT privateHistogram[HISTOGRAM_LENGTH];

    for(int i = tx; i < HISTOGRAM_LENGTH; i += blockDim.x)
    {
        privateHistogram[i] = 0;
    }

    __syncthreads();

    while (idx < height * width)
    {
        atomicAdd(&privateHistogram[grayImage[idx]], 1);
        idx += stride;
    }

    __syncthreads();

    for(int i = tx; i < HISTOGRAM_LENGTH; i += blockDim.x)
    {
        atomicAdd(&histogram[i], privateHistogram[i]);
    }
}

cudaError_t
computeHistogram(const PUCHAR grayImage, PUINT histogram, int height, int width)
{
    dim3 dimBlock(1, 1, 1);
    dim3 dimGrid(1+(width+height-1)/dimBlock.x, 1, 1);

    PUCHAR gpu_grayImage=NULL;
    PUINT  gpu_histogram=NULL;

    wbCheck(cudaHostGetDevicePointer((void **)&gpu_grayImage, (void *)grayImage, 0));
    wbCheck(cudaHostGetDevicePointer((void **)&gpu_histogram, (void *)histogram, 0));

    gpu_computeHistogram<<<dimGrid, dimBlock>>>(gpu_grayImage, gpu_histogram, height, width);

    wbCheck(cudaThreadSynchronize());

    return cudaSuccess;
}

#endif

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

cudaError_t
equalizeHistogram(const float *inputData, float * outputData, int height, int width, int channels)
{
    float  *floatImage = NULL;
    PUCHAR ucharImage  = NULL;
    PUCHAR grayImage   = NULL;
    PUINT  histogram   = NULL;
    float *cdf         = NULL;

    wbCheck(cudaHostAlloc((void **)&floatImage, sizeof(float) * height * width * channels, cudaHostAllocMapped));
    wbCheck(cudaHostAlloc((void **)&ucharImage, sizeof(UCHAR) * height * width * channels, cudaHostAllocMapped));
    wbCheck(cudaHostAlloc((void **)&grayImage, sizeof(UCHAR) * height * width, cudaHostAllocMapped));
    wbCheck(cudaHostAlloc((void **)&histogram, sizeof(UINT) * HISTOGRAM_LENGTH, cudaHostAllocMapped));
    wbCheck(cudaHostAlloc((void **)&cdf, sizeof(float) * HISTOGRAM_LENGTH, cudaHostAllocMapped));

    memcpy(floatImage, inputData, sizeof(float) * height * width * channels);

    wbCheck(castToUChar(floatImage, ucharImage, height, width, channels));
    wbCheck(convertRGBToGray(ucharImage, grayImage, height, width, channels));
    wbCheck(computeHistogram(grayImage, histogram, height, width));
    wbCheck(computeCDF(histogram, cdf, height, width));
    wbCheck(correctColors(ucharImage, cdf, height, width, channels));
    wbCheck(castToFloat(ucharImage, floatImage, height, width, channels));

    memcpy(outputData, floatImage, sizeof(float) * height * width * channels);

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

    cudaDeviceProp prop;
    int whichDevice;

    wbCheck(cudaGetDevice(&whichDevice));
    wbCheck(cudaGetDeviceProperties(&prop, whichDevice));

    wbLog(TRACE, "prop.canMapHostMemory=", prop.canMapHostMemory);

    wbCheck(cudaSetDeviceFlags(cudaDeviceMapHost));
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

