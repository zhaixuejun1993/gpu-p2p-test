
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

    // [GPU Memory Usage] GPU0 36 MB,       GPU1 34 MB    
    cl_mem clbufA_gpu0 = oclctx0.createBuffer3(0, elemCount * sizeof(uint32_t), initBuf0); // create 1G buffer on gpu0 by oclctx0
    // [GPU Memory Usage] GPU0 1061 MB,     GPU1 34 MB 
    /*
    * Question 1: expected GPU0 is 1060 MB (36 + 1024), but is 1061 MB. Why we used 1MB more here?
    */   
    cl_mem clbufB_gpu0 = oclctx0.createBuffer3(0, elemCount * sizeof(uint32_t), initBuf0); // create another 1G buffer on gpu0 by oclctx0
    // [GPU Memory Usage] GPU0 2085 MB,     GPU1 34 MB  
    /*
    * Pass
    */  
    cl_mem clbufC_gpu0 = oclctx0.createBuffer3(0, elemCount * sizeof(uint32_t), initBuf0); // create another 1G buffer on gpu0 by oclctx0
    // [GPU Memory Usage] GPU0 3109 MB,     GPU1 34 MB  
    /*
    * Pass
    */    
    cl_mem clbufD_gpu1 = oclctx0.createBuffer3(1, elemCount * sizeof(uint32_t), initBuf1); // create 1G buffer on gpu1 by oclctx0
    // [GPU Memory Usage] GPU0 3109 MB,     GPU1 1060 MB    
    /*
    * Question 2: Expected GPU1 1058 MB (34 + 1024), but is 1060 MB. Why we used 2MB more here?
    */  
    cl_mem clbufE_gpu1 = oclctx0.createBuffer3(1, elemCount * sizeof(uint32_t), initBuf1); // create another 1G buffer on gpu1 by oclctx0
    // [GPU Memory Usage] GPU0 3110 MB,     GPU1 2084 MB    
    /*
    * Question 3: GPU1 PASS. But why GPU0 got 1MB improve here?
    */ 
    oclctx0.freeBuffer(clbufA_gpu0);
    // [GPU Memory Usage] GPU0 2091 MB,     GPU1 2084 MB   
    /*
    * Question 4: Expected GPU0 2086 MB, but is 2091. Seems there is 5 MB doesn't be released.
    */  
    oclctx0.freeBuffer(clbufB_gpu0);
    // [GPU Memory Usage] GPU0 1061 MB,     GPU1 2084 MB    
    /*
    * Question 4: Expected GPU0 1067 MB, but is 1061. Seems there is another 6 MB be released.
    */
    oclctx0.freeBuffer(clbufC_gpu0);
    // [GPU Memory Usage] GPU0 37 MB,     GPU1 2084 MB   
    /*
    * Pass
    */ 
    oclctx0.freeBuffer(clbufD_gpu1);
    // [GPU Memory Usage] GPU0 37 MB,     GPU1 1060 MB   
    /*
    * Pass
    */  
    oclctx0.freeBuffer(clbufE_gpu1);
    // [GPU Memory Usage] GPU0 37 MB,       GPU1 36 MB  
    /*
    * Question 5: Compare with the bagin status, got 1MB memleak on GPU0 & 2MB on GPU1
    */   
    std::cout << "------ END ------" << std::endl;
    return 0;
}

int main(int argc, char **argv)
{
    ocl_buffer_allocate_behavior();
    return 0;
}