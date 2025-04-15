
#include <CL/cl.h>
#include <iostream>
#include <vector>

#include "ocl_context.h"
#include "lz_context.h"

int ocl_buffer_allocate_behavior()
{
    size_t elemCount = 1024 * 1024 * 256;
    std::vector<uint32_t> initBuf0(elemCount, 0);
    std::vector<uint32_t> initBuf1(elemCount, 1);

    // initialize opencl contexts with 
    oclContext oclctx0;
    oclctx0.init({0, 1});

    // [GPU Memory Usage] GPU0 34 MB,       GPU1 34 MB    
    cl_mem clbufA_gpu0 = oclctx0.createBuffer2(0, elemCount * sizeof(uint32_t), initBuf0); // create 1G buffer on gpu0 by oclctx0
    // [GPU Memory Usage] GPU0 1059 MB,     GPU1 34 MB 
    /*
    * Issue 1: expected GPU0 is 1058 MB (34 + 1024), but used 1059 MB.
    */   
    cl_mem clbufB_gpu0 = oclctx0.createBuffer2(0, elemCount * sizeof(uint32_t), initBuf0); // create another 1G buffer on gpu0 by oclctx0
    // [GPU Memory Usage] GPU0 2083 MB,     GPU1 34 MB  
    /*
    * Pass
    */  
    cl_mem clbufC_gpu0 = oclctx0.createBuffer2(0, elemCount * sizeof(uint32_t), initBuf0); // create another 1G buffer on gpu0 by oclctx0
    // [GPU Memory Usage] GPU0 3107 MB,     GPU1 34 MB  
    /*
    * Pass
    */    
    cl_mem clbufD_gpu1 = oclctx0.createBuffer2(1, elemCount * sizeof(uint32_t), initBuf1); // create 1G buffer on gpu1 by oclctx0
    // [GPU Memory Usage] GPU0 4131 MB,     GPU1 34 MB    
    /*
    * Issue 2: Expected GPU0 no change and GPU1 1058 MB (34 + 1024), but the result show GPU0 improve 1024 MB & GPU1 no change
    */  
    cl_mem clbufE_gpu1 = oclctx0.createBuffer2(1, elemCount * sizeof(uint32_t), initBuf1); // create another 1G buffer on gpu1 by oclctx0
    /*
    * Issue: The same with issue 2
    */ 
    // [GPU Memory Usage] GPU0 5155 MB,     GPU1 34 MB    
    oclctx0.freeBuffer(clbufA_gpu0);
    // [GPU Memory Usage] GPU0 4131 MB,     GPU1 34 MB    
    oclctx0.freeBuffer(clbufB_gpu0);
    // [GPU Memory Usage] GPU0 3107 MB,     GPU1 34 MB    
    oclctx0.freeBuffer(clbufC_gpu0);
    // [GPU Memory Usage] GPU0 2083 MB,     GPU1 34 MB    
    oclctx0.freeBuffer(clbufD_gpu1);
    // [GPU Memory Usage] GPU0 1059 MB,     GPU1 34 MB    
    oclctx0.freeBuffer(clbufE_gpu1);
    // [GPU Memory Usage] GPU0 35 MB,       GPU1 34 MB  
    /*
    * Issue 3: Memleak 1M
    */   
    std::cout << "------ END ------" << std::endl;
    return 0;
}

int main(int argc, char **argv)
{
    ocl_buffer_allocate_behavior();
    return 0;
}