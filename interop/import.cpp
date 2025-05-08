#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <thread>
#include <condition_variable>
#include <random>
#include <atomic>
#include "ocl_context.h"
#include "lz_context.h"

char write_kernel_code[] = " \
kernel void write_to_remote(global int *src1, global int *src2)  \
{ \
  const int id = get_global_id(0); \
  src2[id] = src1[id] * 2; \
} \
";
#define CHECK_ERR(err) \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "[%s] Error %d at %s:%d\n", \
                device_name, err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }
using cl_mem_handle = uint64_t;
typedef struct {
    oclContext oclctx;
    std::string device_name;
    int thread_idx;
    cl_mem peer_mem;
} ThreadData;

typedef struct {
    cl_mem local_mem[2]= {NULL, NULL};
    std::condition_variable cv;
    std::condition_variable import_cv;
    std::mutex mutex;
    size_t sizes[2];
    bool local_ready[2] = {false, false};
    bool import_ready[2] = {false, false};
} SharedState;
SharedState shared;
size_t mem_size = 4* 1024 * 1024;
void* thread_func(ThreadData* data) {
    cl_int err;
    uint64_t export_handle = data->oclctx.deriveHandle(data->peer_mem);
    std::cout << "my GPU " << data->device_name <<  ", local handle: " << export_handle << std::endl;
    cl_mem imported_mem = data->oclctx.createFromHandle(export_handle, mem_size);
    std::cout << "import done" << std::endl;
    auto ret = clReleaseMemObject(imported_mem);
    if (ret != 0)
        std::cout << "release error: " << ret << std::endl;
    return NULL;
}

int main() {

    ThreadData td0, td1;
    td0.oclctx.init({0, 1});
    td1.oclctx.init({1, 0});

    td0.device_name = "GPU0";
    td1.device_name = "GPU1";
    td0.thread_idx = 0;
    td1.thread_idx = 1;
    auto elem_count = mem_size / sizeof(uint32_t);
    std::vector<uint32_t> initBuf(elem_count, 0);
    for (size_t j = 0; j < elem_count; j++)
        initBuf[j] = (j % 1024);  
    cl_mem local_mem_0 = td0.oclctx.createBuffer2(0, mem_size, initBuf);
    cl_mem local_mem_1 = td1.oclctx.createBuffer2(0, mem_size, initBuf);
    td0.peer_mem = local_mem_1;
    td1.peer_mem = local_mem_0;

    std::thread t0(thread_func, &td0);
    std::thread t1(thread_func, &td1);

    t0.join();
    t1.join();


    cl_mem local_mem_2 = td0.oclctx.createBuffer2(0, mem_size, initBuf);
    cl_mem local_mem_3 = td1.oclctx.createBuffer2(0, mem_size, initBuf);
    td0.peer_mem = local_mem_3;
    td1.peer_mem = local_mem_2;

    std::thread t2(thread_func, &td0);
    std::thread t3(thread_func, &td1);

    t2.join();
    t3.join();

    return 0;
}