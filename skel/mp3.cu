// MP 3 -- tiled matrix multipication
#include    <wb.h>

#define TILE_SIZE 16

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int col = blockIdx.x*blockDim.x + tx;
	int row = blockIdx.y*blockDim.y + ty;
	
	float accumulated = 0.0;
	
	__shared__ float tile_A[TILE_SIZE][TILE_SIZE];
	__shared__ float tile_B[TILE_SIZE][TILE_SIZE];
	
	for (int t=0;t < 1+(numAColumns-1)/TILE_SIZE;t++)
	{
		if (row < numARows && t*TILE_SIZE+tx < numAColumns)
		{
			tile_A[ty][tx] = A[row*numAColumns + t*TILE_SIZE + tx];
		}
		else
		{
			tile_A[ty][tx] = 0.0;
		}
											   
		if (col < numBColumns && t*TILE_SIZE+ty < numBRows)
		{
			tile_B[ty][tx] = B[(t*TILE_SIZE+ty)*numBColumns + col];
		}
		else
		{
			tile_B[ty][tx] = 0.0;
		}
		
		__syncthreads();
		
		for(int i=0;i<TILE_SIZE;i++)
		{
			accumulated += tile_A[ty][i] * tile_B[i][tx];
		}
		
		__syncthreads();

	}
	
	if (row < numCRows && col < numCColumns)
	{
		C[row * numCColumns + col] = accumulated;
	}
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");

	hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);

	
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
	hostC = (float *) malloc(numCRows * numCColumns * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
	wbCheck(cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(float)));
	wbCheck(cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
	wbCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice));
	wbCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));
	wbCheck(cudaMemcpy(deviceC, hostC, numCRows * numCColumns * sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
	dim3 DimGrid(1+(numCColumns-1)/TILE_SIZE, 1+(numCRows-1)/TILE_SIZE, 1);
	dim3 DimBlock(TILE_SIZE, TILE_SIZE, 1);
    
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
	matrixMultiplyShared<<<DimGrid, DimBlock>>>(deviceA, deviceB, deviceC,
                   							numARows, numAColumns,
                   							numBRows, numBColumns,
                   							numCRows, numCColumns);


    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
	wbCheck(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
	wbCheck(cudaFree((void *)deviceA));
	wbCheck(cudaFree((void *)deviceB));
	wbCheck(cudaFree((void *)deviceC));
    wbTime_stop(GPU, "Freeing GPU Memory");
	
    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

