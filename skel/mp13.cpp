#include <wb.h> //@@ wb include opencl.h for you

#define BLOCK_SIZE 1024

#define wbCheck(error) do {                                \
        if (error != CL_SUCCESS) {                         \
            wbLog(ERROR, "error = ", error);               \
            return  1;                                     \
        }                                                  \
    } while(0)

//@@ OpenCL Kernel

const char *vaddSrc[] =
{
    "__kernel void vadd(__global const float *a, __global const float *b, __global float *result, unsigned len)",
    "{",
    "   int idx = get_global_id(0);",
    "   if (idx < len)",
    "   {",
    "       result[idx] = a[idx] + b[idx];",
    "   }",
    "}"
}
;

int main(int argc, char **argv) {
    wbArg_t args;
    int inputLength;
    float *hostInput1;
    float *hostInput2;
    float *hostOutput;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *)malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The input length is ", inputLength);

    wbTime_start(GPU, "Inializing OpenCL.");

    cl_int clError = CL_SUCCESS;
    cl_uint clNumPlatforms;

    clError = clGetPlatformIDs(0, NULL, &clNumPlatforms);
    wbCheck(clError);

    cl_platform_id *clPlatforms = (cl_platform_id *)alloca(clNumPlatforms * sizeof(cl_platform_id));
    clError = clGetPlatformIDs(clNumPlatforms, clPlatforms, &clNumPlatforms);
    wbCheck(clError);

    cl_context_properties clContextProperties[] =
    {
        CL_CONTEXT_PLATFORM, (cl_context_properties)clPlatforms[0],
        0
    };

    cl_context clContext = clCreateContextFromType(clContextProperties, CL_DEVICE_TYPE_ALL, NULL, NULL, &clError);
    wbCheck(clError);

    size_t clNumDevices;
    clError = clGetContextInfo(clContext, CL_CONTEXT_DEVICES, 0, NULL, &clNumDevices);
    wbCheck(clError);

    cl_device_id* cldevs = (cl_device_id *)alloca(clNumDevices*sizeof(cl_device_id));

    clError = clGetContextInfo(clContext, CL_CONTEXT_DEVICES, clNumDevices, cldevs, NULL);
    wbCheck(clError);

    cl_command_queue clQueue = clCreateCommandQueue(clContext, cldevs[0], 0, &clError);
    wbCheck(clError);

    cl_program clProgram = clCreateProgramWithSource(clContext, sizeof(vaddSrc)/sizeof(vaddSrc[0]), (const char **)&vaddSrc, NULL, &clError);
    wbCheck(clError);

    char clCompilerFlags[] = "-cl-mad-enable";
    clError = clBuildProgram(clProgram, 0, NULL, clCompilerFlags, NULL, NULL);
    wbCheck(clError);

    cl_kernel clKernel = clCreateKernel(clProgram, "vadd", &clError);
    wbCheck(clError);

    wbTime_stop(GPU, "Inializing OpenCL.");

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cl_mem deviceInput1 = clCreateBuffer(clContext, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, inputLength * sizeof(float), hostInput1, &clError);
    wbCheck(clError);

    cl_mem deviceInput2 = clCreateBuffer(clContext, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, inputLength * sizeof(float), hostInput2, &clError);
    wbCheck(clError);

    cl_mem deviceOutput = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY, inputLength * sizeof(float), NULL, &clError);
    wbCheck(clError);

    wbTime_stop(GPU, "Allocating GPU memory.");

    //@@ Initialize the grid and block dimensions here
    size_t localWorkSize[1] = {BLOCK_SIZE};
    size_t globalWorkSize[1] = {BLOCK_SIZE * ((inputLength + BLOCK_SIZE - 1)/BLOCK_SIZE)};
        
    wbTime_start(Compute, "Performing OpenCL computation");
    //@@ Launch the GPU Kernel here

    clError = clSetKernelArg(clKernel, 0, sizeof(cl_mem), &deviceInput1);
    wbCheck(clError);

    clError = clSetKernelArg(clKernel, 1, sizeof(cl_mem), &deviceInput2);
    wbCheck(clError);

    clError = clSetKernelArg(clKernel, 2, sizeof(cl_mem), &deviceOutput);
    wbCheck(clError);

    clError = clSetKernelArg(clKernel, 3, sizeof(unsigned), &inputLength);
    wbCheck(clError);

    cl_event clEvent = NULL;
    clError = clEnqueueNDRangeKernel(clQueue, clKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, &clEvent);
    wbCheck(clError);

    wbTime_stop(Compute, "Performing OpenCL computation");

    clError = clWaitForEvents(1, &clEvent);
    wbCheck(clError);

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    clError = clEnqueueReadBuffer(clQueue, deviceOutput, CL_TRUE, 0, inputLength*sizeof(float), hostOutput, 0, NULL, NULL);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Resources");
    //@@ Free the GPU memory here

    clReleaseMemObject(deviceOutput);
    clReleaseMemObject(deviceInput2);
    clReleaseMemObject(deviceInput1);

    clReleaseKernel(clKernel);
    clReleaseProgram(clProgram);
    clReleaseCommandQueue(clQueue);
    clReleaseContext(clContext);

    wbTime_stop(GPU, "Freeing GPU Resources");

    wbSolution(args, hostOutput, inputLength);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}
