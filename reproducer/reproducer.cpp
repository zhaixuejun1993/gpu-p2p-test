
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
    oclctx[1].init({1, 0});

    std::vector<uint32_t> initBuf0(elemCount, 0);
    std::vector<uint32_t> initBuf1(elemCount, 1);
    cl_mem org_bufs[2];
    org_bufs[0] = oclctx[0].createBuffer2(0, elemCount * sizeof(uint32_t), initBuf0);
    org_bufs[1] = oclctx[1].createBuffer2(0, elemCount * sizeof(uint32_t), initBuf1);

    cl_mem sub_bufs[2];
    sub_bufs[0] = oclctx[0].createBuffer2(0, elemCount * sizeof(uint32_t), initBuf0);
    sub_bufs[1] = oclctx[1].createBuffer2(0, elemCount * sizeof(uint32_t), initBuf1);

    cl_mem shared_bufs[2];
    uint64_t handle = oclctx[1].deriveHandle(sub_bufs[1]);
    shared_bufs[0] = oclctx[0].createFromHandle(handle, elemCount * sizeof(uint32_t));

    handle = oclctx[0].deriveHandle(sub_bufs[0]);
    shared_bufs[1] = oclctx[1].createFromHandle(handle, elemCount * sizeof(uint32_t));

    auto sub_size = elemCount * sizeof(uint32_t) / 2;
    for (int i = 0; i < 40; i++)
    {
        cl_int ret;

        auto start_time = std::chrono::high_resolution_clock::now();
        ret = clEnqueueCopyBuffer(oclctx[0].queue(), org_bufs[0], shared_bufs[0], sub_size, 0, sub_size, 0, nullptr, nullptr);
        if (ret != CL_SUCCESS)
            std::cout << "Failed!!!" << std::endl;
        auto end_time = std::chrono::high_resolution_clock::now();
        std::cout << "iter: " << i << ", enqueue time: " << std::chrono::duration<double, std::micro>(end_time - start_time).count() << std::endl;
    }

    clReleaseMemObject(shared_bufs[0]);
    clReleaseMemObject(shared_bufs[1]);
    clReleaseMemObject(org_bufs[0]);
    clReleaseMemObject(org_bufs[1]);
    clReleaseMemObject(sub_bufs[0]);
    clReleaseMemObject(sub_bufs[1]);
    return 0;
}

int main(int argc, char **argv)
{
    size_t element_count = 8 * 2048;
    reproducer(element_count);

    return 0;
}