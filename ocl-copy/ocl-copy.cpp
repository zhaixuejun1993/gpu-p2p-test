
#include <CL/cl.h>
#include <iostream>
#include <vector>

#include "ocl_context.h"
#include "lz_context.h"

char read_kernel_code[] = " \
kernel void read_from_remote(global int *src1, global int *src2) \
{ \
  const int id = get_global_id(0); \
  src1[id] = src2[id]; \
} \
";

char write_kernel_code[] = " \
kernel void write_to_remote(global int *src1, global int *src2)  \
{ \
  const int id = get_global_id(0); \
  src2[id] = src1[id]; \
} \
";

int ocl_p2p_sync_multi_devie_ctx_event(int device_0, int device_1, size_t elemCount)
{
    std::vector<uint32_t> initBuf0(elemCount, 0);
    std::vector<uint32_t> initBuf2(elemCount, 2);

    // initialize two opencl contexts
    oclContext oclctx0, oclctx1;
    oclctx0.init({device_0, device_1});
    oclctx1.init({device_1, device_0});

    cl_mem clbuf0 = oclctx0.createBuffer2(0, elemCount * sizeof(uint32_t), initBuf0);
    cl_mem clbuf2 = oclctx1.createBuffer2(1, elemCount * sizeof(uint32_t), initBuf2);
    // oclctx0.printBuffer(clbuf0);
    // oclctx1.printBuffer(clbuf2);

    uint64_t handle2 = oclctx1.deriveHandle(clbuf2);

    cl_mem clbuf2_shared = oclctx0.createFromHandle(handle2, elemCount * sizeof(uint32_t));
    oclctx0.runKernel1(write_kernel_code, "write_to_remote", clbuf0, clbuf2_shared, elemCount);
    // oclctx0.printBuffer(clbuf0);
    // oclctx1.printBuffer(clbuf2);

    oclctx0.freeBuffer(clbuf0);
    oclctx1.freeBuffer(clbuf2);

    return 0;
}

int main(int argc, char **argv)
{
    // ocl_p2p_sync_multi_devie_ctx_event(0, 1, 4096);
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "====== GPU 0 -> GPU 1 ======" << std::endl;
    size_t element_count = 2048;
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
        ocl_p2p_sync_multi_devie_ctx_event(0, 1, element_count);
    }
    std::cout << "====== GPU 1 -> GPU 0 ======" << std::endl;
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
        ocl_p2p_sync_multi_devie_ctx_event(1, 0, element_count);
    }
    return 0;
}