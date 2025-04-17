
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>

#include "ocl_context.h"
#include "lz_context.h"

char copy_kernel_code[] = " \
kernel void copy_two_buf(global int *src, global int *dst, int src_offset, int dst_offset)  \
{ \
  const int id = get_global_id(0); \
  dst[id + dst_offset] = src[id + src_offset]; \
} \
";

const size_t w_size = 2;

double all_reduce_sync_with_event_kernel(int device_0, int device_1, size_t elemCount = 32)
{
  std::vector<uint32_t> initBuf0(elemCount, 1);
  std::vector<uint32_t> initBuf1(elemCount, 2);

  size_t size_in_bytes = elemCount * sizeof(uint32_t);
  // initialize two opencl contexts
  std::vector<oclContext *> contexts;
  oclContext oclctx0, oclctx1;
  oclctx0.init({device_0, device_1});
  oclctx1.init({device_1, device_0});
  contexts.push_back(&oclctx0);
  contexts.push_back(&oclctx1);

  std::vector<cl_mem *> cl_fc_buf;
  cl_mem cl_fc_buf0 = contexts[0]->createBuffer2(0, size_in_bytes, initBuf0); // = fc output buffer in gpu0
  cl_mem cl_fc_buf1 = contexts[1]->createBuffer2(1, size_in_bytes, initBuf1); // = fc output buffer in gpu1
  cl_fc_buf.push_back(&cl_fc_buf0);
  cl_fc_buf.push_back(&cl_fc_buf1);

  std::vector<cl_mem *> cl_sub_bufs(2, nullptr);
  std::vector<cl_mem *> cl_sub_shared_bufs(2, nullptr);
  std::vector<bool> sub_buf_ready_flag(w_size, false);
  size_t sub_elemCount = elemCount / w_size;

  std::atomic<int> copy_ready(0);

  cl_event step1_copy_event[w_size];
  for (int i = 0; i < w_size; i++)
  {
    step1_copy_event[i] = NULL;
  }

  auto task = [&](int w_rank)
  {
    contexts[w_rank]->createCopyKernel(copy_kernel_code, "copy_two_buf");
    size_t sub_size_in_bytes = size_in_bytes / w_size;
    cl_mem cl_sub_buf = contexts[w_rank]->createBuffer2(0, sub_size_in_bytes, {});
    cl_sub_bufs[w_rank] = &cl_sub_buf;
    sub_buf_ready_flag[w_rank] = true;
    cl_int err;

    while (true)
    {
      size_t wait_all_ready = 0;
      for (int idx = 0; idx < static_cast<int>(w_size); idx++)
      {
        if (sub_buf_ready_flag[idx] == true)
          wait_all_ready++;
      }
      if (wait_all_ready == w_size)
      {
        break;
      }
    }

    auto dst_idx = (w_rank + 1) % w_size;
    uint64_t handle_cl_sub_buf_dst = contexts[dst_idx]->deriveHandle(*cl_sub_bufs[dst_idx]);
    cl_mem cl_sub_buf_dst_shared_on_rank = contexts[w_rank]->createFromHandle(handle_cl_sub_buf_dst, sub_size_in_bytes);
    cl_sub_shared_bufs[w_rank] = &cl_sub_buf_dst_shared_on_rank;

    {
      cl_kernel copy_kernel = contexts[w_rank]->copyKernel();

      err = clSetKernelArg(copy_kernel, 0, sizeof(cl_mem), cl_fc_buf[w_rank]);
      CHECK_OCL_ERROR_EXIT(err, "clSetKernelArg failed");

      err = clSetKernelArg(copy_kernel, 1, sizeof(cl_mem), cl_sub_shared_bufs[w_rank]);
      CHECK_OCL_ERROR_EXIT(err, "clSetKernelArg failed");

      int src1_offset = w_rank * sub_elemCount;
      err = clSetKernelArg(copy_kernel, 2, sizeof(int), &src1_offset);
      CHECK_OCL_ERROR_EXIT(err, "clSetKernelArg failed");

      int src2_offset = 0;
      err = clSetKernelArg(copy_kernel, 3, sizeof(int), &src2_offset);
      CHECK_OCL_ERROR_EXIT(err, "clSetKernelArg failed");

      size_t global_size[] = {sub_elemCount};
      std::cout << w_rank << " --- 0.1" << std::endl;
      // err = clEnqueueNDRangeKernel(contexts[w_rank]->queue(), copy_kernel, 1, nullptr, global_size, nullptr, 0, nullptr, nullptr);
      err = clEnqueueNDRangeKernel(contexts[w_rank]->queue(), copy_kernel, 1, nullptr, global_size, nullptr, 0, nullptr, &step1_copy_event[w_rank]);
      std::cout << w_rank << " --- 0.2" << std::endl;
      CHECK_OCL_ERROR_EXIT(err, "clEnqueueNDRangeKernel failed");
      clFinish(contexts[w_rank]->queue());
    }

    copy_ready++;
    while (true)
    {
      if (copy_ready == 2)
      {
        break;
      }
    }

    contexts[dst_idx]->freeBuffer(*cl_sub_shared_bufs[dst_idx]);
    cl_sub_shared_bufs[dst_idx] = nullptr;
    contexts[w_rank]->freeBuffer(*cl_sub_bufs[w_rank]);
    cl_sub_bufs[w_rank] = nullptr;
    clReleaseEvent(step1_copy_event[w_rank]);
  };
  const auto start = std::chrono::high_resolution_clock::now();
  std::thread t0(task, 0);
  std::thread t1(task, 1);
  t0.join();
  t1.join();
  const auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> elapsed = end - start;
  for (int i = 0; i < w_size; i++)
  {
    contexts[i]->freeBuffer(*cl_fc_buf[i]);
  }

  return elapsed.count();
}

int main(int argc, char **argv)
{
  for (int iter = 0; iter < 1000; iter++)
  {
    all_reduce_sync_with_event_kernel(0, 1, 4 * 1024);
  }
  return 0;
}