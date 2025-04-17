
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

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

char add_kernel_code[] = " \
kernel void add_two_buf(global int *src1, global int *src2, int src1_offset, int src2_offset)  \
{ \
  const int id = get_global_id(0); \
  src1[id + src1_offset] = src1[id + src1_offset] + src2[id + src2_offset]; \
} \
";
const size_t w_size = 2;
bool debug_log = false;
double all_reduce_sync_with_host(int device_0, int device_1, size_t elemCount = 32)
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
  if (debug_log)
  {
    contexts[0]->printBuffer(*cl_fc_buf[0], 32);
    contexts[1]->printBuffer(*cl_fc_buf[1], 32);
  }
  std::mutex global_mtx;
  std::vector<cl_mem *> cl_sub_bufs(2, nullptr);
  std::vector<cl_mem *> cl_sub_shared_bufs(2, nullptr);
  std::vector<cl_mem *> cl_sub_shared_bufs_tmp(2, nullptr);
  std::vector<bool> copy_flag(w_size, false);
  std::vector<bool> concat_copy_flag1(w_size, false);
  std::vector<bool> concat_copy_flag2(w_size, false);
  std::vector<bool> sub_buf_ready_flag(w_size, false);
  std::vector<bool> add_flag(w_size, false);
  size_t sub_elemCount = elemCount / w_size;

  std::atomic<int> copy_ready(0);
  std::atomic<int> add_ready(0);
  std::atomic<int> concat_copy_ready1(0);
  std::atomic<int> concat_copy_ready2(0);

  auto task = [&](int w_rank)
  {
    contexts[w_rank]->createAddKernel(add_kernel_code, "add_two_buf");
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

    err = clEnqueueCopyBuffer(contexts[w_rank]->queue(), *cl_fc_buf[w_rank], *cl_sub_shared_bufs[w_rank], w_rank * sub_size_in_bytes, 0, sub_size_in_bytes, 0, nullptr, nullptr);
    CHECK_OCL_ERROR_EXIT(err, "clEnqueueCopyBuffer failed");
    clFinish(contexts[w_rank]->queue());
    copy_flag[w_rank] = true;
    copy_ready++;
    while (true)
    {
      if (copy_ready == 2)
      {
        break;
      }
    }

    if (debug_log)
    {
      std::lock_guard<std::mutex> lock(global_mtx);
      std::cout << "[Rank] " << w_rank << " After copy sub_buf: " << std::endl;
      contexts[w_rank]->printBuffer(*cl_sub_bufs[w_rank]);
    }

    // std::cout << w_rank << " --- 1" << std::endl;

    cl_kernel add_kernel = contexts[w_rank]->addKernel();

    err = clSetKernelArg(add_kernel, 0, sizeof(cl_mem), cl_fc_buf[w_rank]);
    CHECK_OCL_ERROR_EXIT(err, "clSetKernelArg failed");

    err = clSetKernelArg(add_kernel, 1, sizeof(cl_mem), cl_sub_bufs[w_rank]);
    CHECK_OCL_ERROR_EXIT(err, "clSetKernelArg failed");

    int src1_offset = dst_idx * sub_elemCount;
    err = clSetKernelArg(add_kernel, 2, sizeof(int), &src1_offset);
    CHECK_OCL_ERROR_EXIT(err, "clSetKernelArg failed");

    int src2_offset = 0;
    err = clSetKernelArg(add_kernel, 3, sizeof(int), &src2_offset);
    CHECK_OCL_ERROR_EXIT(err, "clSetKernelArg failed");

    size_t global_size[] = {sub_elemCount};
    err = clEnqueueNDRangeKernel(contexts[w_rank]->queue(), add_kernel, 1, nullptr, global_size, nullptr, 0, nullptr, nullptr);
    CHECK_OCL_ERROR_EXIT(err, "clEnqueueNDRangeKernel failed");
    clFinish(contexts[w_rank]->queue());
    add_flag[w_rank] = true;
    add_ready++;
    // std::cout << w_rank << " --- 2" << std::endl;

    while (true)
    {
      if (add_ready == 2)
      {
        break;
      }
    }

    if (debug_log)
    {
      std::lock_guard<std::mutex> lock(global_mtx);
      std::cout << "[Rank] " << w_rank << " After add sub_buf: " << std::endl;
      contexts[w_rank]->printBuffer(*cl_fc_buf[w_rank], 32);
    }

    int32_t sub_part = (w_rank + 1) % w_size;
    int32_t rec_sub_part = (sub_part - 1) < 0 ? (w_size - 1) : (sub_part - 1) % w_size;
    err = clEnqueueCopyBuffer(contexts[w_rank]->queue(), *cl_fc_buf[w_rank], *cl_sub_shared_bufs[w_rank], sub_part * sub_size_in_bytes, 0, sub_size_in_bytes, 0, nullptr, nullptr);
    CHECK_OCL_ERROR_EXIT(err, "clEnqueueCopyBuffer failed");
    clFinish(contexts[w_rank]->queue());
    concat_copy_flag1[w_rank] = true;
    concat_copy_ready1++;
    // std::cout << w_rank << " --- 3" << std::endl;

    while (true)
    {
      if (concat_copy_ready1 == 2)
      {
        break;
      }
    }

    err = clEnqueueCopyBuffer(contexts[w_rank]->queue(), *cl_sub_bufs[w_rank], *cl_fc_buf[w_rank], 0, rec_sub_part * sub_size_in_bytes, sub_size_in_bytes, 0, nullptr, nullptr);
    CHECK_OCL_ERROR_EXIT(err, "clEnqueueCopyBuffer failed");
    clFinish(contexts[w_rank]->queue());
    concat_copy_flag2[w_rank] = true;
    concat_copy_ready2++;
    // std::cout << w_rank << " --- 4" << std::endl;

    while (true)
    {
      if (concat_copy_ready2 == 2)
      {
        break;
      }
    }

    if (debug_log)
    {
      std::lock_guard<std::mutex> lock(global_mtx);
      std::cout << "[Rank] " << w_rank << " After concat sub_buf: " << std::endl;
      contexts[w_rank]->printBuffer(*cl_fc_buf[w_rank], 32);
    }
    // std::cout << w_rank << " --- 5" << std::endl;

    contexts[dst_idx]->freeBuffer(*cl_sub_shared_bufs[dst_idx]);
    cl_sub_shared_bufs[dst_idx] = nullptr;
    contexts[w_rank]->freeBuffer(*cl_sub_bufs[w_rank]);
    cl_sub_bufs[w_rank] = nullptr;
    // std::cout << w_rank << " --- 6" << std::endl;
  };
  const auto start = std::chrono::high_resolution_clock::now();
  std::thread t0(task, 0);
  std::thread t1(task, 1);
  t0.join();
  t1.join();
  const auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> elapsed = end - start;
  // std::cout << "iter: " << iter << " time(ms): " << elapsed.count() << std::endl;
  for (int i = 0; i < w_size; i++)
  {
    contexts[i]->freeBuffer(*cl_fc_buf[i]);
  }

  return elapsed.count();
}

int main(int argc, char **argv)
{
  int iteration = (argc >= 2) ? atoi(argv[1]) : 1;
  debug_log = (argc == 3) ? atoi(argv[2]) : 0;
  size_t element_count = 2048;
  for (int i = 0; i < 15; i++)
  {
    if (debug_log)
      std::cout << "================================================" << std::endl;
    element_count *= 2;
    auto bytes = element_count * sizeof(uint32_t);
    if (bytes / 1024.0 / 1024.0 / 1024.0 >= 1)
      std::cout << "Buffer Size: " << std::setw(8) << bytes / 1024.0 / 1024.0 / 1024.0 << " GB: ";
    else if (bytes / 1024.0 / 1024.0 >= 1)
      std::cout << "Buffer Size: " << std::setw(8) << bytes / 1024.0 / 1024.0 << " MB: ";
    else if (bytes / 1024.0 > 1)
      std::cout << "Buffer Size: " << std::setw(8) << bytes / 1024.0 << " KB: ";
    // std::cout << std::endl;
    double avg_val = 0.0;
    for (int iter = 0; iter < iteration; iter++)
    {
      if (debug_log)
      {
        std::cout << std::endl;
        std::cout << "--- " << iter << " ---" << std::endl;
      }
      avg_val += all_reduce_sync_with_host(0, 1, element_count);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << " avg time(ms): " << avg_val / iteration << std::endl;
  }
  // all_reduce_sync_with_host(0, 1, 1024*1024*256);
  return 0;
}