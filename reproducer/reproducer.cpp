
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <atomic>
#include <thread>
#include "ocl_context.h"

int reproducer(size_t elemCount) {
    // size_t elemCount = 16 * 1024 * 1024;
    uint64_t buf_handle[2];

    oclContext oclctx[2];
    oclctx[0].init({0, 1});

    std::vector<uint32_t> initBuf0(elemCount, 0);
    std::vector<uint32_t> initBuf1(elemCount, 1);

    auto byte_size = elemCount * sizeof(uint32_t);

    cl_mem org_bufs[2];

    org_bufs[0] = oclctx[0].createBuffer2(0, byte_size, initBuf0);

    cl_mem sub_bufs[2];
    sub_bufs[0] = oclctx[0].createBuffer2(0, byte_size, initBuf1);

    cl_mem shared_bufs[2];
    uint64_t handle = oclctx[0].deriveHandle(sub_bufs[0]);
    shared_bufs[0] = oclctx[0].createFromHandle(handle, byte_size);

    cl_int ret;
    for (int i = 0; i < 400; i++)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        ret = clEnqueueCopyBuffer(oclctx[0].queue(), org_bufs[0], shared_bufs[0], 0, 0, byte_size, 0, nullptr, nullptr);
        if (ret != CL_SUCCESS)
            std::cout << "Failed!!!" << std::endl;
        auto end_time = std::chrono::high_resolution_clock::now();
        std::cout << "iter: " << i << ", enqueue time: " << std::chrono::duration<double, std::micro>(end_time - start_time).count() << std::endl;
    }
    clFinish(oclctx[0].queue());

    clReleaseMemObject(shared_bufs[0]);
    clReleaseMemObject(org_bufs[0]);
    clReleaseMemObject(sub_bufs[0]);
    return 0;
}

int main(int argc, char **argv)
{
    size_t element_count = 8000 * 2048;
    reproducer(element_count);

    return 0;
}