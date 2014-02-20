#include	<wb.h>

#define CHUNK_SIZE   64
#define BLOCK_SIZE   16
#define N_STREAMS     4

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

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

	wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here

    float * pinnedHostInput1;
    float * pinnedHostInput2;
    float * pinnedHostOutput;

    wbCheck(cudaHostAlloc((void **)&pinnedHostInput1, inputLength * sizeof(float), cudaHostAllocDefault));
    wbCheck(cudaHostAlloc((void **)&pinnedHostInput2, inputLength * sizeof(float), cudaHostAllocDefault));
    wbCheck(cudaHostAlloc((void **)&pinnedHostOutput, inputLength * sizeof(float), cudaHostAllocDefault));

    wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(Generic, "Creating streams.");

    cudaStream_t streams[N_STREAMS];
    float *deviceInputA[N_STREAMS];
    float *deviceInputB[N_STREAMS];
    float *deviceOutput[N_STREAMS];

    for(int i=0;i<N_STREAMS;i++)
    {
        wbCheck(cudaStreamCreate(&streams[i]));
        wbCheck(cudaMalloc((void **)&deviceInputA[i], CHUNK_SIZE * sizeof(float)));
        wbCheck(cudaMalloc((void **)&deviceInputB[i], CHUNK_SIZE * sizeof(float)));
        wbCheck(cudaMalloc((void **)&deviceOutput[i], CHUNK_SIZE * sizeof(float)));
    }

	wbTime_stop(Generic, "Creating streams.");

    memcpy(pinnedHostInput1, hostInput1, inputLength*sizeof(float));
    memcpy(pinnedHostInput2, hostInput2, inputLength*sizeof(float));

    /*
     * Now the fun begins.  This code assumes that the GPU is capable of running 2 inbound copies, a kernel and an outbound kernel
     * in parallel. 
     * 
     * This requires 4 streams, and in the steady state it is running:
     * - the copyout from iteration      n-3
     * - the kernel from iteration       n-2
     * - the copyin of A for  iteration  n-1
     * - the copyin of B for  iteration  n-0
     * 
     * In order for this to work, we have to "prime the pump" with almost 3 full 
     * iterantions
     */

	wbTime_start(Generic, "Queuing items to streams.");
	wbTime_start(GPU, "Running steams.");

    if (inputLength/CHUNK_SIZE < 4)
    {
        // if there are not 4 iterations worth of work, just queue it all on a single stream -- there isn't enough work
        // to fill the pipeline, so it doesn't really matter how efficiently we do it

        for(int pos=0;pos<inputLength; pos += CHUNK_SIZE)
        {
            int left = min(inputLength-pos, CHUNK_SIZE);
            dim3 DimGrid(1 + (left-1)/BLOCK_SIZE, 1, 1);
            dim3 DimBlock(BLOCK_SIZE, 1, 1);

            wbCheck(cudaMemcpyAsync(deviceInputA[0], pinnedHostInput1+pos, left*sizeof(float), cudaMemcpyHostToDevice, streams[0]));
            wbCheck(cudaMemcpyAsync(deviceInputB[0], pinnedHostInput2+pos, left*sizeof(float), cudaMemcpyHostToDevice, streams[0]));

            vecAdd<<<DimGrid, DimBlock, 0, streams[0]>>>(deviceInputA[0], deviceInputB[0], deviceOutput[0], left);

            wbCheck(cudaMemcpyAsync(pinnedHostOutput+pos, deviceOutput[0], left*sizeof(float), cudaMemcpyDeviceToHost, streams[0]));
        }
    }
    else
    {
        int cur=0;
        int left = CHUNK_SIZE;
        dim3 DimGrid(1 + (left-1)/BLOCK_SIZE, 1, 1);
        dim3 DimBlock(BLOCK_SIZE, 1, 1);

        // pipe fill 0
        wbCheck(cudaMemcpyAsync(deviceInputA[(cur-0)%N_STREAMS], pinnedHostInput1+(cur-0)*CHUNK_SIZE, CHUNK_SIZE*sizeof(float), cudaMemcpyHostToDevice, streams[(cur-0)%N_STREAMS]));
        cur++;

        // pipe fill 1
        wbCheck(cudaMemcpyAsync(deviceInputB[(cur-1)%N_STREAMS], pinnedHostInput2+(cur-1)*CHUNK_SIZE, CHUNK_SIZE*sizeof(float), cudaMemcpyHostToDevice, streams[(cur-1)%N_STREAMS]));
        wbCheck(cudaMemcpyAsync(deviceInputA[(cur-0)%N_STREAMS], pinnedHostInput1+(cur-0)*CHUNK_SIZE, CHUNK_SIZE*sizeof(float), cudaMemcpyHostToDevice, streams[(cur-0)%N_STREAMS]));
        cur++;

        // pipe fill 2
        vecAdd<<<DimGrid, DimBlock, 0, streams[(cur-2)%N_STREAMS]>>>(deviceInputA[(cur-2)%N_STREAMS], deviceInputB[(cur-2)%N_STREAMS], deviceOutput[(cur-2)%N_STREAMS], CHUNK_SIZE);
        wbCheck(cudaMemcpyAsync(deviceInputB[(cur-1)%N_STREAMS], pinnedHostInput2+(cur-1)*CHUNK_SIZE, CHUNK_SIZE*sizeof(float), cudaMemcpyHostToDevice, streams[(cur-1)%N_STREAMS]));
        wbCheck(cudaMemcpyAsync(deviceInputA[(cur-0)%N_STREAMS], pinnedHostInput1+(cur-0)*CHUNK_SIZE, CHUNK_SIZE*sizeof(float), cudaMemcpyHostToDevice, streams[(cur-0)%N_STREAMS]));
        cur++;

        // now the pipe if full, run the loop
        for(;cur<inputLength/CHUNK_SIZE; cur++)
        {

            wbCheck(cudaMemcpyAsync(pinnedHostOutput+(cur-3)*CHUNK_SIZE, deviceOutput[(cur-3)%N_STREAMS], CHUNK_SIZE*sizeof(float), cudaMemcpyDeviceToHost, streams[(cur-3)%N_STREAMS]));
            vecAdd<<<DimGrid, DimBlock, 0, streams[(cur-2)%N_STREAMS]>>>(deviceInputA[(cur-2)%N_STREAMS], deviceInputB[(cur-2)%N_STREAMS], deviceOutput[(cur-2)%N_STREAMS], CHUNK_SIZE);
            wbCheck(cudaMemcpyAsync(deviceInputB[(cur-1)%N_STREAMS], pinnedHostInput2+(cur-1)*CHUNK_SIZE, CHUNK_SIZE*sizeof(float), cudaMemcpyHostToDevice, streams[(cur-1)%N_STREAMS]));

            left = min(inputLength-(cur*CHUNK_SIZE), CHUNK_SIZE);

            wbCheck(cudaMemcpyAsync(deviceInputA[(cur-0)%N_STREAMS], pinnedHostInput1+(cur-0)*CHUNK_SIZE, left*sizeof(float), cudaMemcpyHostToDevice, streams[(cur-0)%N_STREAMS]));
        }

        // empty the pipe -- this will take 3 steps too (just like filling it)
        // empty step 1
        wbCheck(cudaMemcpyAsync(pinnedHostOutput+(cur-3)*CHUNK_SIZE, deviceOutput[(cur-3)%N_STREAMS], CHUNK_SIZE*sizeof(float), cudaMemcpyDeviceToHost, streams[(cur-3)%N_STREAMS]));
        vecAdd<<<DimGrid, DimBlock, 0, streams[(cur-2)%N_STREAMS]>>>(deviceInputA[(cur-2)%N_STREAMS], deviceInputB[(cur-2)%N_STREAMS], deviceOutput[(cur-2)%N_STREAMS], CHUNK_SIZE);
        wbCheck(cudaMemcpyAsync(deviceInputB[(cur-1)%N_STREAMS], pinnedHostInput2+(cur-1)*CHUNK_SIZE, left*sizeof(float), cudaMemcpyHostToDevice, streams[(cur-1)%N_STREAMS]));
        cur++;

        // empty step 2
        DimGrid.x = 1 + (left-1)/BLOCK_SIZE;
        wbCheck(cudaMemcpyAsync(pinnedHostOutput+(cur-3)*CHUNK_SIZE, deviceOutput[(cur-3)%N_STREAMS], CHUNK_SIZE*sizeof(float), cudaMemcpyDeviceToHost, streams[(cur-3)%N_STREAMS]));
        vecAdd<<<DimGrid, DimBlock, 0, streams[(cur-2)%N_STREAMS]>>>(deviceInputA[(cur-2)%N_STREAMS], deviceInputB[(cur-2)%N_STREAMS], deviceOutput[(cur-2)%N_STREAMS], left);
        cur++;

        // empty step 3
        wbCheck(cudaMemcpyAsync(pinnedHostOutput+(cur-3)*CHUNK_SIZE, deviceOutput[(cur-3)%N_STREAMS], left*sizeof(float), cudaMemcpyDeviceToHost, streams[(cur-3)%N_STREAMS]));
        cur++;
    }

	wbTime_stop(Generic, "Queuing items to streams.");

	wbTime_start(Generic, "Synchronizing for streams.");
    for(int i=0;i<N_STREAMS;i++)
    {
        wbCheck(cudaStreamSynchronize(streams[i]));
    }
	wbTime_stop(Generic, "Synchronizing for streams.");

	wbTime_stop(GPU, "Running steams.");

    memcpy(hostOutput, pinnedHostOutput, inputLength*sizeof(float));

    wbSolution(args, hostOutput, inputLength);

    wbCheck(cudaFreeHost(pinnedHostInput1));
    wbCheck(cudaFreeHost(pinnedHostInput2));
    wbCheck(cudaFreeHost(pinnedHostOutput));

    for(int i=0;i<N_STREAMS;i++)
    {
        wbCheck(cudaStreamDestroy(streams[i]));
        wbCheck(cudaFree(deviceInputA[i]));
        wbCheck(cudaFree(deviceInputB[i]));
        wbCheck(cudaFree(deviceOutput[i]));
    }

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

