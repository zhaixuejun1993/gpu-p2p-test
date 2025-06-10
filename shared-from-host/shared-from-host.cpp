
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <atomic>
#include <thread>
#include <string>
#include <iomanip>
#include "ocl_context.h"

int reproducer(size_t elemCount)
{
    auto byte_size = elemCount * sizeof(uint32_t);
    cl_int ret;
    oclContext oclctx;
    oclctx.init({0, 1});
    auto print_value = [&](std::string name, cl_mem &buf0, cl_mem &buf1)
    {
        std::cout << name << ": " << std::endl;
        std::vector<uint32_t> result0(elemCount, 2);
        std::vector<uint32_t> result1(elemCount, 3);
        ret = clEnqueueReadBuffer(oclctx.queue(), buf0, CL_TRUE, 0, byte_size, result0.data(), 0, NULL, NULL);
        ret = clEnqueueReadBuffer(oclctx.queue1(), buf1, CL_TRUE, 0, byte_size, result1.data(), 0, NULL, NULL);
        for (int i = 0; i < 32; i++)
            std::cout << result0[i] << ", ";
        std::cout << std::endl;
        for (int i = 0; i < 32; i++)
            std::cout << result1[i] << ", ";
        std::cout << std::endl;
    };

    std::vector<uint32_t> initBuf0(elemCount, 2);
    std::vector<uint32_t> initBuf1(elemCount, 3);

    cl_mem org_bufs[2];
    org_bufs[0] = oclctx.createBuffer2(0, byte_size, initBuf0);
    org_bufs[1] = oclctx.createBuffer3(1, byte_size, initBuf1);
    print_value("org_bufs", org_bufs[0], org_bufs[1]);

    cl_mem shared_buffers[2];
    shared_buffers[0] = clCreateBuffer(
        oclctx.context(),
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        byte_size,
        nullptr,
        &ret);
    shared_buffers[1] = clCreateBuffer(
        oclctx.context(),
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        byte_size,
        nullptr,
        &ret);
    print_value("shared_buffers", shared_buffers[0], shared_buffers[1]);

    cl_event copy_to_shared_event[2];
    ret = clEnqueueCopyBuffer(oclctx.queue(), org_bufs[0], shared_buffers[0], 0, 0, byte_size, 0, nullptr, &copy_to_shared_event[0]);
    ret = clEnqueueCopyBuffer(oclctx.queue1(), org_bufs[1], shared_buffers[1], 0, 0, byte_size, 0, nullptr, &copy_to_shared_event[1]);
    // print_value("shared_buffers after copy", shared_buffers[0], shared_buffers[1]);

    ret = clEnqueueCopyBuffer(oclctx.queue(), shared_buffers[1], org_bufs[0], 0, 0, byte_size, 2, copy_to_shared_event, nullptr);
    ret = clEnqueueCopyBuffer(oclctx.queue1(), shared_buffers[0], org_bufs[1], 0, 0, byte_size, 2, copy_to_shared_event, nullptr);

    clFinish(oclctx.queue());
    clFinish(oclctx.queue1());
    print_value("org_bufs after copy", org_bufs[0], org_bufs[1]);

    clReleaseMemObject(org_bufs[0]);
    clReleaseMemObject(org_bufs[1]);
    clReleaseMemObject(shared_buffers[0]);
    clReleaseMemObject(shared_buffers[1]);
    return 0;
}

int main(int argc, char **argv)
{
    size_t element_count = 8000 * 2048;
    reproducer(element_count);

    element_count = 2048;
    for (int i = 0; i < 17; i++)
    {
        element_count *= 2;
        auto bytes = element_count * sizeof(uint32_t);
        if (bytes / 1024.0 / 1024.0 / 1024.0 >= 1)
            std::cout << "BW [GBPS]: " << std::setw(8) << bytes / 1024.0 / 1024.0 / 1024.0 << " GB: ";
        else if (bytes / 1024.0 / 1024.0 >= 1)
            std::cout << "BW [GBPS]: " << std::setw(8) << bytes / 1024.0 / 1024.0 << " MB: ";
        else if (bytes / 1024.0 > 1)
            std::cout << "BW [GBPS]: " << std::setw(8) << bytes / 1024.0 << " KB: ";
        reproducer(element_count);
        std::cout << std::endl;
    }

    return 0;
}