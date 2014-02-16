// MP 6 Scan // Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}
#include    <wb.h>

#define BLOCK_SIZE 256

#define wbCheck(stmt) do {                                 \
        cudaError_t err = stmt;                            \
        if (err != cudaSuccess) {                          \
            wbLog(ERROR, "Failed to run stmt ", #stmt);    \
            return 0;                                      \
        }                                                  \
    } while(0)

#ifdef TEST_ON_HOST
void scan(float * input, float * output, int len)
{
    double sum = 0.0;

    for(int i=0;i<len;i++)
    {
        sum += input[i];
        output[i] = sum;
    }
}
#else
__global__ void addOffset(float *vector, float *offsets, int len) {

    int tx = threadIdx.x;
    int idx = blockIdx.x*blockDim.x + tx;

    __shared__ float offset;

    if (blockIdx.x > 0)
    {
        if (tx == 0)
        {
            offset = offsets[blockIdx.x-1];
        }

        __syncthreads();

        if (idx < len)
        {
            vector[idx] += offset;
        }
    }
}

__global__ void doscan(float * input, float * output, float *blockTotal, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here

    __shared__ float XY[2*BLOCK_SIZE];
    int tx = threadIdx.x;
    int idx = blockIdx.x*blockDim.x + tx;

    // load two values into shared memory

    if (idx < len)
    {
        XY[tx] = input[idx];
    }
    else
    {
        XY[tx] = 0;
    }

    if (idx + BLOCK_SIZE < len)
    {
        XY[tx+BLOCK_SIZE] = input[idx+BLOCK_SIZE];
    }
    else
    {
        XY[tx+BLOCK_SIZE] = 0;
    }
	

    // reduction phase
    for(int stride=1;stride <= BLOCK_SIZE; stride *= 2)
    {
        int idx = (tx + 1)*stride*2 - 1;

        __syncthreads();
        if (idx < 2*BLOCK_SIZE)
        {
            XY[idx] += XY[idx-stride];
        }
    }

    // reverse phase
    for(int stride=BLOCK_SIZE/2;stride>0;stride /= 2)
    {
        int idx = (tx + 1)*stride*2 - 1;

		__syncthreads();
        if (idx + stride < 2*BLOCK_SIZE)
        {
            XY[idx + stride] += XY[idx];
        }
    }
    
    // write output after all the threads are done
    __syncthreads();
    if (idx < len)
    {
        output[idx] = XY[tx];
    }

    if (tx == (blockDim.x - 1) || idx == (len - 1) )
    {
        blockTotal[blockIdx.x] = XY[tx];
    }
}

int scan(float * input, float * output, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here

    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
	dim3 DimGrid(1+(len-1)/BLOCK_SIZE, 1, 1);
	dim3 DimBlock(BLOCK_SIZE, 1, 1);

    float *hostBlockTotal = NULL;
    float *deviceBlockTotal = NULL;

    hostBlockTotal = (float *)malloc(DimGrid.x * sizeof(float));
    wbCheck(cudaMalloc((void**)&deviceBlockTotal, DimGrid.x*sizeof(float)));

    doscan<<<DimGrid, DimBlock>>>(input, output, deviceBlockTotal, len);
    wbCheck(cudaDeviceSynchronize());

    if (DimGrid.x > 1)
    {
        // we have to deal with multiple blocks

        int nextLen = DimGrid.x;

        float *deviceOutput = NULL;

        wbCheck(cudaMalloc((void**)&deviceOutput, nextLen*sizeof(float)));

        scan(deviceBlockTotal, deviceOutput, nextLen);

        // now update the blocks to reflect the offsets

        addOffset<<<DimGrid, DimBlock>>>(output, deviceOutput, len);
        wbCheck(cudaDeviceSynchronize());

        wbCheck(cudaFree((void*)deviceOutput));
    }

    free(hostBlockTotal);
    wbCheck(cudaFree(deviceBlockTotal));

    return 0;
}
#endif

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

#ifdef TEST_ON_HOST
    scan(hostInput, hostOutput, numElements);
#else
    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    wbTime_start(Compute, "Performing CUDA computation");
    scan(deviceInput, deviceOutput, numElements);
    wbCheck(cudaDeviceSynchronize());
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");
#endif

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}


