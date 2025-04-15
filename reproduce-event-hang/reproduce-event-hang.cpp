
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

#include "ocl_context.h"
#include "lz_context.h"

const size_t w_size = 2;
bool debug_log = true;
void test_fun0(size_t elemCount = 32)
{
  std::vector<uint32_t> initBuf0(elemCount, 1);
  std::vector<uint32_t> initBuf1(elemCount, 2);

  // initialize two opencl contexts
  std::vector<oclContext *> contexts;
  oclContext oclctx0, oclctx1;
  oclctx0.init({0, 1});
  oclctx1.init({1, 0});
  contexts.push_back(&oclctx0);
  contexts.push_back(&oclctx1);

  size_t size_in_bytes = elemCount * sizeof(uint32_t);
  std::vector<cl_mem *> cl_fc_buf;
  cl_mem cl_fc_buf0 = contexts[0]->createBuffer2(0, size_in_bytes, initBuf0);
  cl_mem cl_fc_buf1 = contexts[1]->createBuffer2(1, size_in_bytes, initBuf1);
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
  std::vector<bool> sub_buf_ready_flag(w_size, false);
  std::vector<bool> ready_free_flag(w_size, false);
  std::vector<cl_event> copy_event(w_size, NULL);

  auto task = [&](int w_rank)
  {
    cl_mem cl_sub_buf = contexts[w_rank]->createBuffer2(0, size_in_bytes, {});

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
    cl_mem cl_sub_buf_dst_shared_on_rank = contexts[w_rank]->createFromHandle(handle_cl_sub_buf_dst, size_in_bytes);
    cl_sub_shared_bufs[w_rank] = &cl_sub_buf_dst_shared_on_rank;

    err = clEnqueueCopyBuffer(contexts[w_rank]->queue(), *cl_fc_buf[w_rank], *cl_sub_shared_bufs[w_rank], 0, 0, size_in_bytes, 0, nullptr, &copy_event[w_rank]);
    CHECK_OCL_ERROR_EXIT(err, "clEnqueueCopyBuffer failed");

    while (true)
    {
      if (copy_event[dst_idx] != NULL)
        break;
    }
    err = clWaitForEvents(1, &copy_event[dst_idx]);
    CHECK_OCL_ERROR_EXIT(err, "clEnqueueCopyBuffer failed");
    if (debug_log)
    {
      std::lock_guard<std::mutex> lock(global_mtx);
      std::cout << "========================" << std::endl;
      std::cout << "[Step 1][Rank] " << w_rank << " sub_buf: " << std::endl;
      contexts[w_rank]->printBuffer(*cl_sub_bufs[w_rank]);
    }

    ready_free_flag[w_rank] = true;
    while (true)
    {
      size_t wait_all_ready = 0;
      for (int idx = 0; idx < static_cast<int>(w_size); idx++)
      {
        if (ready_free_flag[idx] == true)
          wait_all_ready++;
      }
      if (wait_all_ready == w_size)
      {
        break;
      }
    }
    contexts[w_rank]->freeBuffer(*cl_sub_shared_bufs[w_rank]);
    cl_sub_shared_bufs[w_rank] = nullptr;
    contexts[w_rank]->freeBuffer(*cl_sub_bufs[w_rank]);
    cl_sub_bufs[w_rank] = nullptr;
    clReleaseEvent(copy_event[w_rank]);
    copy_event[w_rank] = NULL;
  };
  const auto start = std::chrono::high_resolution_clock::now();
  std::thread t0(task, 0);
  std::thread t1(task, 1);
  t0.join();
  t1.join();
  for (int i = 0; i < w_size; i++)
  {
    contexts[i]->freeBuffer(*cl_fc_buf[i]);
  }
}

int main(int argc, char **argv)
{
  int iteration = (argc >= 2) ? atoi(argv[1]) : 1;
  debug_log = (argc == 3) ? atoi(argv[2]) : 0;
  size_t element_count = 2048;
  for (int i = 0; i < 10; i++)
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
      test_fun0(element_count);
    }
    std::cout << " avg time(ms): " << avg_val / iteration << std::endl;
  }

  return 0;
}