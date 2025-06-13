
// #include <CL/cl.h>
#include <iostream>
#include <vector>
#include <atomic>
#include <thread>
#include <string>
#include <iomanip>
#include "ocl_context.h"
#include <CL/opencl.hpp>

int main(int argc, char **argv)
{
    cl_int error_code;

    cl_uint num_platforms = 0;
    error_code = clGetPlatformIDs(0, NULL, &num_platforms);
    if (num_platforms != 1)
        std::cout << "multi platform!" << std::endl;

    std::vector<cl_platform_id> platform_ids(num_platforms);
    error_code = clGetPlatformIDs(num_platforms, platform_ids.data(), NULL);

    cl::Context ctx;
    cl::Platform platform = cl::Platform(platform_ids[0]);
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    ctx = cl::Context(devices);

    std::string kernel_code = "__kernel void test() {}";
    cl::Program::Sources sources;
    sources.push_back({kernel_code.c_str(), kernel_code.length()});

    if (devices.size() != 2)
        std::cout << "Can not find 2 gpu device!" << std::endl;

    cl::Program program1(ctx, sources);
    program1.build({devices[1]});
    cl::vector<cl::Kernel> kernels;
    program1.createKernels(&kernels);
    std::cout << "create kernel1 size: " << kernels.size() << std::endl;

    // cl::Program program2(ctx, sources);
    // program2.build({devices[1]});
    // cl::vector<cl::Kernel> kernels2;
    // program2.createKernels(&kernels2);
    // std::cout << "create kernel2 size: " << kernels2.size() << std::endl;

    return 0;
}