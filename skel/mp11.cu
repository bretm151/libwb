// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256

const float WEIGHT_R = 0.21;
const float WEIGHT_G = 0.71;
const float WEIGHT_B = 0.07;

typedef unsigned char UCHAR, *PUCHAR;
typedef unsigned int UINT, *PUINT;

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return err;                                    \
        }                                                  \
    } while(0)

//@@ insert code here

#ifdef USE_HOST
#else
#endif

void
castToUChar(const float *input, PUCHAR output, int height, int width, int channels)
{
    for(int i=0;i<height*width*channels;i++)
    {
        output[i] = (unsigned char)(255.0 * input[i]);
    }
}

void
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
}
void
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
}

void
computeCDF(const PUINT histogram, float *cdf, int height, int width)
{
    float prev = 0.0;

    for(int i=0;i<HISTOGRAM_LENGTH;i++)
    {
        cdf[i] = prev + histogram[i]/(float)(height*width);
        prev = cdf[i];
    }
}

UCHAR
clamp(const UCHAR x, const UCHAR start, const UCHAR end)
{
    return min(max(x,start), end);
}

UCHAR
correctColor(UCHAR val, float *cdf)
{
    return clamp(255*(cdf[val]-cdf[0])/(1-cdf[0]), 0, 255);
}

void
correctColors(PUCHAR ucharImage, float *cdf, int height, int width, int channels)
{
    for(int i=0;i<height*width*channels;i++)
    {
        ucharImage[i] = correctColor(ucharImage[i], cdf);
    }
}

void
castToFloat(const PUCHAR input, float *output, int height, int width, int channels)
{
    for(int i=0;i<height*width*channels;i++)
    {
        output[i] = (float)(input[i]/255.0);
    }
}

cudaError_t
equalizeHistogram(int height, int width, int channels, const float *inputData, float * outputData)
{
    PUCHAR ucharImage = (PUCHAR) malloc(sizeof(UCHAR) * height * width * channels);
    PUCHAR grayImage  = (PUCHAR) malloc(sizeof(UCHAR) * height * width * channels);
    PUINT  histogram  = (PUINT)  malloc(sizeof(UINT)  * HISTOGRAM_LENGTH);
    float *cdf        = (float*) malloc(sizeof(float) * HISTOGRAM_LENGTH);

    castToUChar(inputData, ucharImage, height, width, channels);
    convertRGBToGray(ucharImage, grayImage, height, width, channels);
    computeHistogram(grayImage, histogram, height, width);
    computeCDF(histogram, cdf, height, width);
    correctColors(ucharImage, cdf, height, width, channels);
    castToFloat(ucharImage, outputData, height, width, channels);

    free(grayImage);
    free(ucharImage);

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

    wbCheck(equalizeHistogram(imageHeight, 
                              imageWidth,
                              imageChannels,
                              wbImage_getData(inputImage),
                              wbImage_getData(outputImage)));

    wbSolution(args, outputImage);

    //@@ insert code here
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}

