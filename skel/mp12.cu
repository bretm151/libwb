#include	<wb.h>

#define CHUNK_SIZE  512
#define BLOCK_SIZE   64

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", err);                        \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (idx < len)
	{
		out[idx] = in1[idx] + in2[idx];
	}
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;

    float * pinnedHostInput1;
    float * pinnedHostInput2;
    float * pinnedHostOutput;

    float * deviceInput0_1;
    float * deviceInput0_2;
    float * deviceOutput0;

    float * deviceInput1_1;
    float * deviceInput1_2;
    float * deviceOutput1;

    float * deviceInput2_1;
    float * deviceInput2_2;
    float * deviceOutput2;

    float * deviceInput3_1;
    float * deviceInput3_2;
    float * deviceOutput3;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

	wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here

    wbCheck(cudaHostAlloc((void **)&pinnedHostInput1, inputLength * sizeof(float), cudaHostAllocDefault));
    wbCheck(cudaHostAlloc((void **)&pinnedHostInput2, inputLength * sizeof(float), cudaHostAllocDefault));
    wbCheck(cudaHostAlloc((void **)&pinnedHostOutput, inputLength * sizeof(float), cudaHostAllocDefault));

	wbCheck(cudaMalloc((void **)&deviceInput0_1, CHUNK_SIZE * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceInput0_2, CHUNK_SIZE * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceOutput0, CHUNK_SIZE * sizeof(float)));

	wbCheck(cudaMalloc((void **)&deviceInput1_1, CHUNK_SIZE * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceInput1_2, CHUNK_SIZE * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceOutput1, CHUNK_SIZE * sizeof(float)));

	wbCheck(cudaMalloc((void **)&deviceInput2_1, CHUNK_SIZE * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceInput2_2, CHUNK_SIZE * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceOutput2, CHUNK_SIZE * sizeof(float)));

	wbCheck(cudaMalloc((void **)&deviceInput3_1, CHUNK_SIZE * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceInput3_2, CHUNK_SIZE * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceOutput3, CHUNK_SIZE * sizeof(float)));

    wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(Generic, "Creating streams.");

    cudaStream_t stream0;
    cudaStream_t stream1;
    cudaStream_t stream2;
    cudaStream_t stream3;

    wbCheck(cudaStreamCreate(&stream0));
    wbCheck(cudaStreamCreate(&stream1));
    wbCheck(cudaStreamCreate(&stream2));
    wbCheck(cudaStreamCreate(&stream3));

	wbTime_stop(Generic, "Creating streams.");

    memcpy(pinnedHostInput1, hostInput1, inputLength*sizeof(float));
    memcpy(pinnedHostInput2, hostInput2, inputLength*sizeof(float));

	wbTime_start(Generic, "Queuing items to streams.");
	wbTime_start(GPU, "Running steams.");
    for(int pos=0;pos<inputLength; pos += CHUNK_SIZE)
    {
        int left = min(inputLength-pos, CHUNK_SIZE);
        dim3 DimGrid(1 + (left-1)/BLOCK_SIZE, 1, 1);
        dim3 DimBlock(BLOCK_SIZE, 1, 1);

        wbCheck(cudaMemcpyAsync(deviceInput0_1, pinnedHostInput1+pos, left*sizeof(float), cudaMemcpyHostToDevice, stream0));
        wbCheck(cudaMemcpyAsync(deviceInput0_2, pinnedHostInput2+pos, left*sizeof(float), cudaMemcpyHostToDevice, stream0));

        vecAdd<<<DimGrid, DimBlock, 0, stream0>>>(deviceInput0_1, deviceInput0_2, deviceOutput0, left);

        wbCheck(cudaMemcpyAsync(pinnedHostOutput+pos, deviceOutput0, left*sizeof(float), cudaMemcpyDeviceToHost, stream0));
    }
	wbTime_stop(Generic, "Queuing items to streams.");

	wbTime_start(Generic, "Synchronizing for streams.");
    wbCheck(cudaStreamSynchronize(stream0));
    wbCheck(cudaStreamSynchronize(stream1));
    wbCheck(cudaStreamSynchronize(stream2));
    wbCheck(cudaStreamSynchronize(stream3));
	wbTime_stop(Generic, "Synchronizing for streams.");

	wbTime_stop(GPU, "Running steams.");

    memcpy(hostOutput, pinnedHostOutput, inputLength*sizeof(float));

    wbSolution(args, hostOutput, inputLength);

    wbCheck(cudaFreeHost(pinnedHostInput1));
    wbCheck(cudaFreeHost(pinnedHostInput2));
    wbCheck(cudaFreeHost(pinnedHostOutput));

	wbCheck(cudaFree(deviceInput0_1));
	wbCheck(cudaFree(deviceInput0_2));
	wbCheck(cudaFree(deviceOutput0));

	wbCheck(cudaFree(deviceInput1_1));
	wbCheck(cudaFree(deviceInput1_2));
	wbCheck(cudaFree(deviceOutput1));

	wbCheck(cudaFree(deviceInput2_1));
	wbCheck(cudaFree(deviceInput2_2));
	wbCheck(cudaFree(deviceOutput2));

	wbCheck(cudaFree(deviceInput3_1));
	wbCheck(cudaFree(deviceInput3_2));
	wbCheck(cudaFree(deviceOutput3));

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

